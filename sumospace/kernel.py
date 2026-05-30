# sumospace/kernel.py

"""
SumoKernel — Main Orchestration Engine
========================================
The kernel is the entry point for all task execution.

Pipeline:
  1. Classify intent (3-stage hybrid classifier)
  2. Retrieve context via RAG (if needed)
  3. Committee deliberation → approved execution plan
  4. Execute plan step-by-step via ToolRegistry
  5. Stream results + persist to memory

Usage:
    kernel = SumoKernel()                         # Zero config, no API key
    kernel = SumoKernel(KernelConfig(provider="ollama"))
    kernel = SumoKernel(KernelConfig(provider="gemini", model="gemini-1.5-flash"))

    async with kernel:
        trace = await kernel.run("Refactor the auth module in src/auth.py")
        print(trace.final_answer)
"""

from __future__ import annotations

import time
import uuid
import hashlib
import json
import re
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from sumospace.settings import SumoSettings

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Any, AsyncIterator

from sumospace.audit import AuditLogger
from sumospace.cache import PlanCache
from sumospace.classifier import ClassificationResult, Intent, SumoClassifier
from sumospace.telemetry import SumoTelemetry
from sumospace.committee import Committee, CommitteeVerdict, ExecutionPlan
from sumospace.exceptions import (
    ConsensusFailedError,
    ExecutionHaltedError,
    KernelBootError,
)
from sumospace.hooks import HookRegistry
from sumospace.ingest import UniversalIngestor
from sumospace.memory import MemoryManager
from sumospace.providers import ProviderRouter
from sumospace.rag import RAGPipeline
from sumospace.scope import ScopeManager
from sumospace.settings import SumoSettings
from sumospace.templates import TemplateManager
from sumospace.tools import ToolRegistry, ToolResult

console = Console()


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class KernelConfig:
    """
    Runtime configuration for a kernel instance.
    All defaults work with zero API keys.

    For cloud providers, set provider + matching API key in environment.
    """
    # ── Model provider ────────────────────────────────────────────────────────
    provider: str = "hf"           # hf | ollama | auto | gemini | openai | anthropic
    model: str = "default"         # "default" resolves per-provider

    # ── Embeddings (local by default) ─────────────────────────────────────────
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ── Execution control ─────────────────────────────────────────────────────
    require_consensus: bool = True
    max_retries: int = 3
    execution_timeout: int = 120
    verbose: bool = True
    dry_run: bool = False
    hf_load_in_4bit: bool = False

    # ── Secondary model (for planning/structured tasks) ──────────────────────
    secondary_provider: str | None = None
    secondary_model: str | None = None

    # ── Paths ─────────────────────────────────────────────────────────────────
    workspace: str = "."

    # ── Scope & isolation ─────────────────────────────────────────────────────
    scope_level: str = "user"         # user | session | project
    user_id: str = ""                 # Already-validated user identifier
    session_id: str = ""              # Session identifier (for session-level scope)
    project_id: str = ""              # Project identifier (for project-level scope)
    chroma_base: str = ".sumo_db"     # Base directory for scoped DB paths
    chroma_path: str = ""             # Deprecated — use chroma_base
    max_chunks_per_scope: int | None = None   # Quota guard per scope (None = unlimited)

    def __post_init__(self):
        if self.chroma_path and self.chroma_base == ".sumo_db":
            import warnings
            warnings.warn(
                "chroma_path is deprecated, use chroma_base instead.",
                DeprecationWarning, stacklevel=2
            )
            self.chroma_base = self.chroma_path


# ─── Execution Trace ──────────────────────────────────────────────────────────

@dataclass
class StepTrace:
    step_number: int
    tool: str
    description: str
    result: ToolResult
    duration_ms: float
    thought: str = ""
    parameters: dict = field(default_factory=dict)
    estimated_tokens: int = 0
    provider_tokens: int = 0
    snapshot_context: str = ""

from enum import Enum

class FailureCategory(str, Enum):
    ROUTING_FAILURE = "routing_failure"
    PARSING_FAILURE = "parsing_failure"
    INVALID_EDIT = "invalid_edit"
    HALLUCINATED_TOOL = "hallucinated_tool"
    CRITIC_DEADLOCK = "critic_deadlock"
    CONTEXT_OVERFLOW = "context_overflow"
    TIMEOUT = "timeout"
    MALFORMED_PARAMS = "malformed_params"
    UNKNOWN = "unknown"

@dataclass
class SynthesisChunk:
    delta: str


@dataclass
class ExecutionTrace:
    task: str
    session_id: str
    intent: Intent
    classification: ClassificationResult
    plan: ExecutionPlan | None
    step_traces: list[StepTrace] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    error: str = ""
    failure_category: FailureCategory | None = None
    duration_ms: float = 0.0
    retries: int = 0
    total_estimated_tokens: int = 0
    total_provider_tokens: int = 0
    rag_context: str = ""

    @property
    def tool_outputs(self) -> list[str]:
        return [t.result.output for t in self.step_traces]

    @property
    def failed_steps(self) -> list[StepTrace]:
        return [t for t in self.step_traces if not t.result.success]

    def to_json(self) -> dict:
        import json
        from dataclasses import asdict
        return json.loads(
            json.dumps(asdict(self), default=str)
        )


# ─── Kernel ───────────────────────────────────────────────────────────────────

class SumoKernel:
    """
    The main orchestration engine.

    Lifecycle:
        async with SumoKernel() as kernel:
            trace = await kernel.run("your task")
    """

    def __init__(
        self,
        config: KernelConfig | None = None,
        settings: "SumoSettings | None" = None,
        hooks: "HookRegistry | None" = None,
        domain_context: "DomainContext | None" = None,
    ):
        if config is not None and settings is None:
            import warnings
            warnings.warn(
                "Passing KernelConfig directly is deprecated. "
                "Use SumoSettings instead: SumoKernel(settings=SumoSettings(...)). "
                "KernelConfig support will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            import dataclasses
            from sumospace.settings import SumoSettings
            data = dataclasses.asdict(config)
            # Map deprecated flags to new settings
            if "require_consensus" in data:
                data["committee_enabled"] = data["require_consensus"]
            if "dry_run" in data:
                data["execution_enabled"] = not data["dry_run"]
            
            self.settings = SumoSettings(**data)
        elif settings is not None:
            self.settings = settings
        else:
            from sumospace.settings import SumoSettings
            self.settings = SumoSettings()

        self._provider: ProviderRouter | None = None
        self._classifier: SumoClassifier | None = None
        self._committee: Committee | None = None
        self._tools: ToolRegistry | None = None
        self._memory: MemoryManager | None = None
        self._ingestor: UniversalIngestor | None = None
        self._rag: RAGPipeline | None = None
        self._initialized = False

        from sumospace.audit import AuditLogger
        self._audit_logger: AuditLogger | None = AuditLogger(self.settings)

        from sumospace.hooks import HookRegistry
        self.hooks: HookRegistry = hooks or HookRegistry(verbose=self.settings.verbose)

        from sumospace.templates import TemplateManager
        self.templates = TemplateManager(
            template_path=self.settings.prompt_template_path
        )

        # Auto-load hooks from workspace if enabled
        self._auto_load_hooks()

        from sumospace.cache import PlanCache
        self._cache = PlanCache(
            cache_dir=str(Path(self.settings.workspace) / ".sumo_cache")
        )

        self.telemetry = SumoTelemetry(
            enabled=self.settings.telemetry_enabled,
            endpoint=self.settings.telemetry_endpoint
        )

        # Domain context: explicit object takes priority, then settings fields
        if domain_context is not None:
            self._domain_context = domain_context
        else:
            from sumospace.domain_context import DomainContext
            cfg = self.settings
            has_any = any([
                getattr(cfg, 'global_domain_context', ''),
                getattr(cfg, 'planner_domain_context', ''),
                getattr(cfg, 'critic_domain_context', ''),
                getattr(cfg, 'resolver_domain_context', ''),
            ])
            if has_any:
                self._domain_context = DomainContext(
                    global_context=getattr(cfg, 'global_domain_context', ''),
                    planner_context=getattr(cfg, 'planner_domain_context', ''),
                    critic_context=getattr(cfg, 'critic_domain_context', ''),
                    resolver_context=getattr(cfg, 'resolver_domain_context', ''),
                )
            else:
                self._domain_context = None

    @property
    def tools(self) -> ToolRegistry:
        """Access the internal ToolRegistry."""
        if self._tools is None:
            raise RuntimeError("Kernel not booted. Call boot() or use async with first.")
        return self._tools

    async def boot(self):
        """Initialise all subsystems. Called automatically by async context manager."""
        if self._initialized:
            return

        cfg = self.settings
        try:
            if cfg.verbose:
                console.print(Panel(
                    f"[bold cyan]SumoKernel booting[/bold cyan]\n"
                    f"Provider: [green]{cfg.provider}[/green]  "
                    f"Model: [green]{cfg.model}[/green]  "
                    f"Embeddings: [green]{cfg.embedding_provider}[/green]\n"
                    f"Workspace: [dim]{cfg.workspace}[/dim]  "
                    f"Dry-run: [yellow]{cfg.dry_run}[/yellow]",
                    title="SumoSpace",
                    border_style="cyan",
                ))

            # 1. Provider
            self._provider = ProviderRouter(
                provider=cfg.provider,
                model=cfg.model if cfg.model != "default" else None,
                load_in_4bit=cfg.hf_load_in_4bit,
            )
            await self._provider.initialize()

            self._secondary_provider = None
            if cfg.secondary_provider:
                self._secondary_provider = ProviderRouter(
                    provider=cfg.secondary_provider,
                    model=cfg.secondary_model,
                    load_in_4bit=cfg.hf_load_in_4bit,
                )
                await self._secondary_provider.initialize()

            # 2. Tool registry
            from sumospace.snapshots import SnapshotManager
            self._snapshot_manager = SnapshotManager(cfg)
            self._tools = ToolRegistry(workspace=cfg.workspace, snapshot_manager=self._snapshot_manager)

            # 3. Scope resolution
            #    If user_id is set, build a ScopeManager and resolve paths.
            #    Otherwise fall back to raw chroma_base.
            scope_mgr = None
            resolved_chroma = cfg.chroma_base
            if cfg.user_id:
                from sumospace.scope import ScopeManager
                scope_mgr = ScopeManager(
                    chroma_base=cfg.chroma_base,
                    level=cfg.scope_level,
                )
                resolved_chroma = scope_mgr.resolve(
                    user_id=cfg.user_id,
                    session_id=cfg.session_id,
                    project_id=cfg.project_id,
                )

            # 4. Memory
            if cfg.memory_enabled:
                self._memory = MemoryManager(
                    chroma_path=resolved_chroma,
                    embedding_provider=cfg.embedding_provider,
                    scope_manager=scope_mgr,
                    user_id=cfg.user_id,
                    session_id=cfg.session_id,
                    project_id=cfg.project_id,
                )
                await self._memory.initialize()
            else:
                self._memory = None

            # 5. Ingestor + RAG
            if getattr(cfg, "rag_enabled", True):
                from sumospace.vectorstores import get_vector_store
                vs = get_vector_store(cfg)
                
                self._ingestor = UniversalIngestor(
                    vector_store=vs,
                    chroma_path=resolved_chroma,
                    embedding_provider=cfg.embedding_provider,
                    embedding_model=cfg.embedding_model,
                    max_chunks=cfg.max_chunks_per_scope,
                )
                await self._ingestor.initialize()

                self._rag = RAGPipeline(ingestor=self._ingestor)
                await self._rag.initialize()
            else:
                self._ingestor = None
                self._rag = None

            # 5. Classifier
            self._classifier = SumoClassifier(provider=self._provider)
            await self._classifier.initialize()

            # 6. Committee
            self._committee = Committee(
                provider=self._provider,
                planning_provider=self._secondary_provider or self._provider,
                require_consensus=cfg.require_consensus,
                templates=self.templates,
                domain_context=self._domain_context,
            )

            self._initialized = True

            if cfg.verbose:
                console.print("[bold green]✓ Kernel ready[/bold green]")

            await self.hooks.trigger("on_kernel_boot", self)

        except Exception as e:
            raise KernelBootError(f"Kernel boot failed: {e}") from e

    async def shutdown(self):
        """Graceful shutdown."""
        await self.hooks.trigger("on_kernel_shutdown", self)

        self._initialized = False

        if self._memory and hasattr(self._memory, 'episodic'):
            try:
                client = self._memory.episodic._client
                if client and hasattr(client, '_system'):
                    client._system.stop()
            except Exception:
                pass

        if getattr(self, '_ingestor', None) is not None:
            try:
                client = self._ingestor._client
                if client and hasattr(client, '_system'):
                    client._system.stop()
            except Exception:
                pass

        try:
            import chromadb
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        if cfg := self.settings:
            if cfg.verbose:
                console.print("[dim]Kernel shutdown[/dim]")

    async def __aenter__(self):
        await self.boot()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()

    # ── Main Entry Point ─────────────────────────────────────────────────────

    async def run(self, task: str, session_id: str | None = None) -> ExecutionTrace:
        """
        Execute a task end-to-end synchronously.

        Args:
            task:       Natural language task description.
            session_id: Optional session identifier for memory scoping.

        Returns:
            ExecutionTrace with full audit trail and final answer.

        Note:
            Prefer `stream_run()` over this method in any UI context. `run()` blocks
            until full completion, meaning the user will see no feedback during
            potentially long-running tool executions or committee deliberation.

        Warning:
            If you catch `ConsensusFailedError` or `ExecutionHaltedError`, the returned 
            trace will have `success=False` and the error attached to `trace.error`.
        """
        if not self._initialized:
            await self.boot()

        session_id = session_id or uuid.uuid4().hex[:12]
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:8]
        start = time.monotonic()
        verdict = None

        async with self.telemetry.async_span(
            "sumospace.kernel.run", 
            attributes={"task": task, "session_id": session_id, "task_hash": task_hash}
        ):
            trace = ExecutionTrace(
                task=task,
                session_id=session_id,
                intent=Intent.GENERAL_QA,
                classification=None,
                plan=None,
            )

        try:
            # Step 1: Classify
            if self.settings.verbose:
                console.print(f"\n[bold]Task:[/bold] {task}")

            await self.hooks.trigger("on_task_start", task, session_id)

            recent_ctx = {}
            if self.settings.memory_enabled:
                recent_ctx["recent_messages"] = [m["content"] for m in self._memory.recent(5)]

            async with self.telemetry.async_span("sumospace.classify", attributes={"task": task}):
                classification = await self._classifier.classify(task, context=recent_ctx)
            trace.intent = classification.intent
            trace.classification = classification

            if self.settings.verbose:
                console.print(
                    f"[dim]Intent: [cyan]{classification.intent.value}[/cyan] "
                    f"({classification.confidence:.0%}) — {classification.reasoning}[/dim]"
                )

            # Step 2: RAG retrieval (if needed)
            rag_context = ""
            if self.settings.rag_enabled and classification.needs_retrieval:
                async with self.telemetry.async_span("sumospace.rag.retrieve", attributes={"task": task}):
                    try:
                        rag_result = await self._rag.retrieve(task)
                        if rag_result.chunks:
                            rag_context = rag_result.context
                            trace.rag_context = rag_context
                            if self.settings.verbose:
                                console.print(
                                    f"[dim]Retrieved {len(rag_result.chunks)} chunks "
                                    f"(reranked: {rag_result.used_reranker})[/dim]"
                                )
                    except Exception as e:
                        if self.settings.verbose:
                            console.print(f"[yellow]RAG skipped: {e}[/yellow]")

            # Step 3: Web search (if needed)
            web_context = ""
            if classification.needs_web:
                async with self.telemetry.async_span("sumospace.web_search", attributes={"task": task}):
                    web_result = await self._tools.execute("web_search", run_id=session_id, query=task)
                if web_result.success:
                    web_context = web_result.output

            # Build full context
            full_context = self._build_full_context(
                task=task,
                rag_context=rag_context,
                web_context=web_context,
                memory_str=self._memory.context_string(5) if self.settings.memory_enabled and self._memory.recent(1) else ""
            )

            # Direct Inference Bypass — only used when NOT in ReAct mode.
            # In ReAct mode, the agent must use tools even without a committee.
            if not self.settings.committee_enabled and self.settings.execution_mode != "react":
                if self.settings.verbose:
                    console.print("[dim]Committee disabled — direct inference[/dim]")
                prompt = f"{task}\n\nContext:\n{rag_context}" if rag_context else task
                answer = await self._provider.complete(
                    user=prompt,
                    system=self.templates.get("system_prompt"),
                    temperature=self.settings.committee_temperature,
                    max_tokens=self.settings.committee_max_tokens,
                )
                trace.final_answer = answer
                if self.settings.dry_run or not self.settings.execution_enabled:
                    prefix = "[DRY RUN]" if self.settings.dry_run else "[EXECUTION DISABLED]"
                    trace.final_answer = f"{prefix} {trace.final_answer}"
                trace.success = True
                trace.plan = None
                
                if self.settings.memory_enabled:
                    await self._memory.add("user", task)
                    await self._memory.add("assistant", trace.final_answer)
                
                trace.duration_ms = (time.monotonic() - start) * 1000
                if self._audit_logger:
                    self._audit_logger.log(trace, verdict=None)
                await self.hooks.trigger("on_task_complete", trace)
                return trace

            # ReAct without committee: execute autonomously with no pre-built plan.
            if not self.settings.committee_enabled:
                if self.settings.verbose:
                    console.print("[dim]Committee disabled — ReAct autonomous execution[/dim]")
                from sumospace.schemas import ExecutionPlan as _EP
                empty_plan = _EP(steps=[], goal=task, estimated_duration_s=0.0, reasoning="Autonomous execution")
                if self.settings.dry_run or not self.settings.execution_enabled:
                    trace.final_answer = (
                        f"[{'DRY RUN' if self.settings.dry_run else 'EXECUTION DISABLED'}] "
                        f"Would execute autonomously: {task}"
                    )
                    trace.success = True
                else:
                    await self._execute_react(task, empty_plan, trace, full_context)
                if not trace.final_answer:
                    answer_parts = []
                    async for chunk in self._synthesise(task, trace, full_context):
                        answer_parts.append(chunk)
                    trace.final_answer = "".join(answer_parts)
                if self.settings.memory_enabled:
                    await self._memory.add("user", task)
                    await self._memory.add("assistant", trace.final_answer)
                trace.success = True
                trace.plan = None
                trace.duration_ms = (time.monotonic() - start) * 1000
                if self._audit_logger:
                    self._audit_logger.log(trace, verdict=None)
                await self.hooks.trigger("on_task_complete", trace)
                return trace

            # Step 5: Committee deliberation
            cached_plan = self._cache.get(task, full_context)
            if cached_plan:
                if self.settings.verbose:

                    console.print("[dim]Using cached execution plan[/dim]")
                verdict = CommitteeVerdict(
                    approved=True, plan=cached_plan, rejection_reason="", 
                    planner_output="CACHED", critic_output="CACHED", resolver_output="CACHED"
                )
            else:
                if self.settings.verbose:
                    console.print("[dim]Committee deliberating...[/dim]")

                async with self.telemetry.async_span("sumospace.committee.deliberate", attributes={"task": task, "committee.mode": self.settings.committee_mode, "committee.enabled": self.settings.committee_enabled}):
                    verdict = await self._committee.deliberate(task, context=full_context, mode=self.settings.committee_mode)
                
                if verdict.approved:
                    self._cache.set(task, full_context, verdict.plan)
                    
            trace.plan = verdict.plan

            if not verdict.approved:
                await self.hooks.trigger("on_plan_rejected", verdict.rejection_reason, verdict)
                raise ConsensusFailedError(f"Committee rejected plan: {verdict.rejection_reason}")

            await self.hooks.trigger("on_plan_approved", verdict.plan, verdict)

            if self.settings.verbose:
                console.print(
                    f"[green]✓ Plan approved[/green] — "
                    f"{len(verdict.plan.steps)} steps, "
                    f"~{verdict.plan.estimated_duration_s:.0f}s estimated"
                )

            # Step 6: Execute (skip if dry_run)
            if self.settings.dry_run:
                trace.final_answer = self._format_dry_run(verdict)
                trace.success = True
            elif self.settings.execution_mode == "react":
                async with self.telemetry.async_span("sumospace.execute.react", attributes={"task": task}):
                    await self._execute_react(task, verdict.plan, trace, full_context)
            elif self.settings.execution_mode == "plan_execute":
                async with self.telemetry.async_span("sumospace.execute.plan_execute", attributes={"task": task}):
                    await self._execute_plan_and_execute(task, verdict.plan, trace, full_context)

            # Step 7: Synthesise final answer
            if not trace.final_answer:
                answer_parts = []
                async for chunk in self._synthesise(task, trace, full_context):
                    answer_parts.append(chunk)
                trace.final_answer = "".join(answer_parts)

            # Step 8: Persist to memory
            if self.settings.memory_enabled:
                await self._memory.add("user", task)
                await self._memory.add("assistant", trace.final_answer)

            
            trace.success = True

        except ConsensusFailedError as e:
            trace.error = str(e)
            trace.success = False
            trace.failure_category = FailureCategory.CRITIC_DEADLOCK
            trace.final_answer = f"Task halted: {e}"
            if self.settings.verbose:
                console.print(f"[red]✗ {e}[/red]")

        except ExecutionHaltedError as e:
            trace.error = str(e)
            trace.success = False
            trace.failure_category = FailureCategory.INVALID_EDIT
            trace.final_answer = f"Execution halted at critical step: {e}"
            if self.settings.verbose:
                console.print(f"[red]✗ {e}[/red]")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            trace.failure_category = FailureCategory.UNKNOWN
            trace.final_answer = f"Unexpected error: {e}"
            if self.settings.verbose:
                console.print_exception()

        trace.duration_ms = (time.monotonic() - start) * 1000

        if self.settings.verbose:
            status = "[green]✓ Done[/green]" if trace.success else "[red]✗ Failed[/red]"
            console.print(
                f"{status} in {trace.duration_ms:.0f}ms — "
                f"{len(trace.step_traces)} steps executed"
            )

        if self._audit_logger:
            self._audit_logger.log(trace, verdict)

        if trace.success:
            await self.hooks.trigger("on_task_complete", trace)
        else:
            await self.hooks.trigger("on_task_failed", trace, trace.error)

        return trace

    async def stream_run(
        self, task: str, session_id: str | None = None
    ) -> AsyncIterator[StepTrace | SynthesisChunk | ExecutionTrace]:
        """
        Stream execution step-by-step incrementally.

        Args:
            task:       Natural language task description.
            session_id: Optional session identifier for memory scoping.

        Yields:
            `StepTrace` as each tool finishes executing.
            `SynthesisChunk` for partial output of the final answer generation.
            `ExecutionTrace` exactly once at the end.

        Note:
            Prefer this over `run()` in any UI context. `run()` blocks until
            full completion; `stream_run()` lets you show progress incrementally.

        Warning:
            The final yielded object is `ExecutionTrace`, not `StepTrace`.
            Always check `isinstance(event, ExecutionTrace)` to detect completion
            and retrieve the overall `success` status and `final_answer`.
        """
        if not self._initialized:
            await self.boot()

        session_id = session_id or uuid.uuid4().hex[:12]
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:8]
        start = time.monotonic()
        
        async with self.telemetry.async_span(
            "sumospace.kernel.stream_run", 
            attributes={"task": task, "session_id": session_id, "task_hash": task_hash}
        ):
            trace = ExecutionTrace(
                task=task,
                session_id=session_id,
                intent=Intent.GENERAL_QA,
                classification=None,
                plan=None,
            )

        try:
            await self.hooks.trigger("on_task_start", task, session_id)

            recent_ctx = {}
            if self.settings.memory_enabled:
                recent_ctx["recent_messages"] = [m["content"] for m in self._memory.recent(5)]
            async with self.telemetry.async_span("sumospace.classify", attributes={"task": task}):
                classification = await self._classifier.classify(task, context=recent_ctx)
            trace.intent = classification.intent
            trace.classification = classification

            rag_context = ""
            if self.settings.rag_enabled and classification.needs_retrieval:
                async with self.telemetry.async_span("sumospace.rag.retrieve", attributes={"task": task}):
                    try:
                        rag_result = await self._rag.retrieve(task)
                        if rag_result.chunks:
                            rag_context = rag_result.context
                            trace.rag_context = rag_context
                    except Exception:
                        pass

            web_context = ""
            if classification.needs_web:
                async with self.telemetry.async_span("sumospace.web_search", attributes={"task": task}):
                    web_result = await self._tools.execute("web_search", run_id=trace.session_id, query=task)
                if web_result.success:
                    web_context = web_result.output

            # Build full context
            full_context = self._build_full_context(
                task=task,
                rag_context=rag_context,
                web_context=web_context,
                memory_str=self._memory.context_string(5) if self.settings.memory_enabled and self._memory.recent(1) else ""
            )

            # Direct Inference Bypass
            if not self.settings.committee_enabled:
                prompt = f"{task}\n\nContext:\n{rag_context}" if rag_context else task
                answer_parts = []
                async for chunk in self._provider.stream(
                    user=prompt,
                    system=self.templates.get("system_prompt"),
                    temperature=self.settings.committee_temperature,
                    max_tokens=self.settings.committee_max_tokens,
                ):
                    answer_parts.append(chunk)
                    yield SynthesisChunk(chunk)
                
                trace.final_answer = "".join(answer_parts)
                if self.settings.dry_run or not self.settings.execution_enabled:
                    prefix = "[DRY RUN]" if self.settings.dry_run else "[EXECUTION DISABLED]"
                    trace.final_answer = f"{prefix} {trace.final_answer}"
                trace.success = True
                trace.plan = None
                
                if self.settings.memory_enabled:
                    await self._memory.add("user", task)
                    await self._memory.add("assistant", trace.final_answer)
                
                trace.duration_ms = (time.monotonic() - start) * 1000
                if self._audit_logger:
                    self._audit_logger.log(trace, verdict=None)
                await self.hooks.trigger("on_task_complete", trace)
                yield trace
                return

            cached_plan = self._cache.get(task, full_context)
            if cached_plan:
                verdict = CommitteeVerdict(
                    approved=True, plan=cached_plan, rejection_reason="", 
                    planner_output="CACHED", critic_output="CACHED", resolver_output="CACHED"
                )
            else:
                async with self.telemetry.async_span("sumospace.committee.deliberate", attributes={"task": task, "committee.mode": self.settings.committee_mode, "committee.enabled": self.settings.committee_enabled}):
                    verdict = await self._committee.deliberate(task, context=full_context, mode=self.settings.committee_mode)
                if verdict.approved:
                    self._cache.set(task, full_context, verdict.plan)

            trace.plan = verdict.plan

            if not verdict.approved:
                await self.hooks.trigger("on_plan_rejected", verdict.rejection_reason, verdict)
                trace.error = verdict.rejection_reason
                trace.success = False
                trace.final_answer = f"Task halted: Committee rejected plan: {verdict.rejection_reason}"
                trace.duration_ms = (time.monotonic() - start) * 1000
                if self._audit_logger:
                    self._audit_logger.log(trace, verdict)
                await self.hooks.trigger("on_task_failed", trace, trace.error)
                yield trace
                return

            await self.hooks.trigger("on_plan_approved", verdict.plan, verdict)

            if self.settings.dry_run or not self.settings.execution_enabled:
                trace.final_answer = self._format_dry_run(verdict)
                if not self.settings.execution_enabled:
                    trace.final_answer = trace.final_answer.replace("[DRY RUN]", "[EXECUTION DISABLED]")
                trace.success = True
            else:
                for step in verdict.plan.steps:
                    await self.hooks.trigger("on_step_start", step)
                    step_start = time.monotonic()
                    result = await self._tools.execute(step.tool, run_id=trace.session_id, **step.parameters)
                    step_ms = (time.monotonic() - step_start) * 1000

                    step_trace = StepTrace(
                        step_number=step.step_number,
                        tool=step.tool,
                        description=step.description,
                        result=result,
                        duration_ms=step_ms,
                    )
                    trace.step_traces.append(step_trace)
                    
                    if result.success:
                        await self.hooks.trigger("on_step_complete", step_trace)
                    else:
                        await self.hooks.trigger("on_step_failed", step_trace)

                    yield step_trace

                    if not result.success and step.critical:
                        raise ExecutionHaltedError(f"Step {step.step_number} ({step.tool}) failed")

            if not trace.final_answer:
                answer_parts = []
                async for chunk in self._synthesise(task, trace, full_context):
                    answer_parts.append(chunk)
                    yield SynthesisChunk(delta=chunk)
                trace.final_answer = "".join(answer_parts)

            if self.settings.memory_enabled:
                await self._memory.add("user", task)
                await self._memory.add("assistant", trace.final_answer)
            trace.success = True

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            trace.final_answer = f"Error: {e}"

        trace.duration_ms = (time.monotonic() - start) * 1000
        if self._audit_logger:
            self._audit_logger.log(trace, locals().get("verdict"))

        if trace.success:
            await self.hooks.trigger("on_task_complete", trace)
        else:
            await self.hooks.trigger("on_task_failed", trace, trace.error)

        yield trace

    # ── ReAct Execution ────────────────────────────────────────────────────────

    def _build_tool_schemas(self, available_tool_names: list[str]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["schema"]
                }
            }
            for t in self._tools.list_tools() if t["name"] in available_tool_names
        ]

    async def _execute_react(self, task: str, plan: ExecutionPlan, trace: ExecutionTrace, context: str):
        """
        ReAct (Reason+Act) execution loop using native tool calling API.
        """
        import json

        if not self.settings.execution_enabled:
            trace.final_answer = (
                f"[Execution disabled] Plan has {len(plan.steps)} steps:\n" +
                "\n".join(f"  {i+1}. {s.tool}: {s.description}" for i, s in enumerate(plan.steps))
            )
            trace.success = True
            return

        self._steps_executed = 0
        step_num = 0
        max_steps = self.settings.react_max_steps

        tools_list = [t for t in self._tools.list_tools() if t['name'] != 'invalid_tool']
        available_tool_names = [t['name'] for t in tools_list]

        plan_outline = "\n".join([
            f"  {s.step_number}. {s.tool}: {s.description}"
            for s in plan.steps
        ])

        from sumospace.context import ContextManager, StepRecord
        ctx = ContextManager(workspace_root=self.settings.workspace)
        
        system = (
            "You are an autonomous coding agent. You have access to tools to read, write, "
            "and modify files. You MUST use the provided tools to complete this task. "
            "Do NOT answer conversationally. Do NOT explain what you will do. "
            "Call a tool immediately."
        )

        initial_context = f"TASK: {task}\nAPPROVED PLAN (use as guidance):\n{plan_outline}" if plan_outline else f"TASK: {task}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": initial_context}
        ]

        MIN_STEPS_BEFORE_FINISH = 1 if getattr(trace.classification, "needs_execution", True) else 0

        tool_signatures = []

        for iteration in range(max_steps):
            messages = await self._trim_messages(messages, max_chars=80000)
            
            filtered_names = ctx.filter_tools(available_tool_names)
            schemas = self._build_tool_schemas(filtered_names)
            
            if self.settings.verbose:
                console.print(f"  [cyan][ReAct {iteration+1}/{max_steps}][/cyan] Thinking...")

            try:
                response = await self._provider.complete_with_tools(
                    messages=messages,
                    tools=schemas
                )
            except NotImplementedError as e:
                import sys
                print(f"[FATAL] Provider does not support tool calling: {e}", file=sys.stderr)
                if self.settings.verbose:
                    console.print(f"    [red]✗ Provider {self._provider.name} does not support complete_with_tools[/red]")
                trace.success = False
                trace.error = str(e)
                break
            except Exception as e:
                if self.settings.verbose:
                    console.print(f"    [red]✗ LLM error: {e}[/red]")
                break

            if response is None:
                trace.success = False
                trace.error = "Provider returned empty or malformed response."
                if self.settings.verbose:
                    console.print(f"    [red]✗ {trace.error}[/red]")
                break

            if response.get("type") == "tool_calls":
                tool_calls = response["tool_calls"]
                # Append the raw provider-specific assistant message containing the tool calls
                if "assistant_message" in response:
                    messages.append(response["assistant_message"])
                else:
                    # Fallback if provider didn't return one (should not happen now)
                    if response.get("content"):
                        messages.append({"role": "assistant", "content": response.get("content")})
                    messages.append({"role": "assistant", "content": f"Calling tools: {', '.join([tc['name'] for tc in tool_calls])}"})

                for tc in tool_calls:
                    tool_name = tc["name"]
                    parameters = tc["arguments"]
                    
                    if isinstance(parameters, str):
                        import json
                        try:
                            parameters = json.loads(parameters)
                        except json.JSONDecodeError as e:
                            error_msg = f"Error: malformed tool arguments — {e}. Try again with valid JSON."
                            tool_call_id = tc.get("id", "")
                            tool_msg = self._provider.format_tool_result(tool_call_id, tool_name, error_msg)
                            messages.append(tool_msg)
                            if self.settings.verbose:
                                console.print(f"    [red]✗ {error_msg}[/red]")
                            await self.hooks.trigger("on_step_failed", None, error_msg)
                            continue

                    self._steps_executed += 1
                    step_num += 1
                    
                    sig_hash = hash(json.dumps(parameters, sort_keys=True))
                    signature = f"{tool_name}:{sig_hash}"
                    tool_signatures.append(signature)
                    
                    repeats = tool_signatures.count(signature)
                    if repeats == 5:
                        raise ExecutionHaltedError(f"Repetition detected: {tool_name} called with identical arguments 5 times.")
                    elif repeats >= 3:
                        if self.settings.verbose:
                            console.print(f"    [yellow]⚠ Repetition Warning ({repeats}/5)[/yellow]")
                        messages.append({
                            "role": "user", 
                            "content": f"[SYSTEM WARNING] You are repeating the same action ({tool_name}). Please try a different approach."
                        })
                    
                    if self.settings.verbose:
                        import json
                        console.print(
                            f"  [cyan][{step_num}][/cyan] "
                            f"{tool_name}: {json.dumps(parameters)[:120]}"
                        )
                        
                    await self.hooks.trigger("on_step_start", None)
                    step_start = time.monotonic()
                    
                    try:
                        import asyncio
                        if self.settings.committee_enabled and getattr(self._committee, "_critic", None):
                            try:
                                hook_result = await asyncio.wait_for(
                                    self._committee._critic.evaluate_tool_call(
                                        tool_name=tool_name,
                                        tool_args=parameters,
                                        messages_history=messages,
                                        safety_context=None
                                    ),
                                    timeout=15.0
                                )
                                if hook_result.action == "reject":
                                    error_msg = hook_result.rejection_message or "Tool call rejected by safety critic."
                                    tool_call_id = tc.get("id", "")
                                    tool_msg = self._provider.format_tool_result(tool_call_id, tool_name, error_msg)
                                    messages.append(tool_msg)
                                    if self.settings.verbose:
                                        console.print(f"    [yellow]⚠ Critic Rejected:[/yellow] {error_msg}")
                                    continue
                                elif hook_result.action == "mutate" and hook_result.mutated_args:
                                    parameters = hook_result.mutated_args
                                    if self.settings.verbose:
                                        console.print(f"    [yellow]⚠ Critic Mutated arguments[/yellow]")
                            except asyncio.TimeoutError:
                                if self.settings.verbose:
                                    console.print("    [yellow]⚠ Critic evaluation timed out, defaulting to approve.[/yellow]")
                            except Exception as e:
                                if self.settings.verbose:
                                    console.print(f"    [yellow]⚠ Critic evaluation failed ({e}), defaulting to approve.[/yellow]")

                        async with self.telemetry.async_span(f"sumospace.react.{tool_name}"):
                            result = await asyncio.wait_for(
                                self._tools.execute(tool_name, run_id=trace.session_id, **parameters),
                                timeout=30.0
                            )
                        
                        step_ms = (time.monotonic() - step_start) * 1000
                        step_trace = StepTrace(step_number=step_num, tool=tool_name, description=f"ReAct step: {tool_name}", result=result, duration_ms=step_ms, thought="", parameters=parameters, estimated_tokens=0, snapshot_context="")
                        trace.step_traces.append(step_trace)
                        
                        tool_call_id = tc.get("id", "")
                        content_str = result.output if result.success else f"Error: {result.error}"
                        tool_msg = self._provider.format_tool_result(tool_call_id, tool_name, content_str)
                        messages.append(tool_msg)
                        
                        if result.success:
                            if self.settings.verbose:
                                preview = result.output[:120].replace("\n", " ") if result.output else "(ok)"
                                console.print(f"    [green]✓[/green] {preview}{'...' if result.output and len(result.output) > 120 else ''}")
                            await self.hooks.trigger("on_step_complete", step_trace)
                        else:
                            if self.settings.verbose:
                                console.print(f"    [red]✗ {result.error}[/red]")
                            await self.hooks.trigger("on_step_failed", None, result.error)
                            
                        if "path" in parameters:
                            ctx.add_active_file(parameters["path"])
                            
                    except asyncio.TimeoutError:
                        error_msg = f"Tool execution timed out after 30 seconds."
                        tool_call_id = tc.get("id", "")
                        tool_msg = self._provider.format_tool_result(tool_call_id, tool_name, error_msg)
                        messages.append(tool_msg)
                        if self.settings.verbose:
                            console.print(f"    [red]✗ {error_msg}[/red]")
                        await self.hooks.trigger("on_step_failed", None, error_msg)
                    except Exception as e:
                        error_msg = f"Tool execution failed: {e}"
                        tool_call_id = tc.get("id", "")
                        tool_msg = self._provider.format_tool_result(tool_call_id, tool_name, error_msg)
                        messages.append(tool_msg)
                        if self.settings.verbose:
                            console.print(f"    [red]✗ {error_msg}[/red]")
                        await self.hooks.trigger("on_step_failed", None, error_msg)

                if any(f.endswith(".py") for f in ctx.active_files):
                    ctx.symbol_graph.update_index()
                
                ctx_update = f"[ENVIRONMENT UPDATE]\nActive Files: {', '.join(ctx.active_files) if ctx.active_files else '(None)'}\nWorkspace Symbols:\n{ctx.symbol_graph.format_summary()}"
                messages.append({"role": "user", "content": ctx_update})

            else:
                if self._steps_executed < MIN_STEPS_BEFORE_FINISH:
                    messages.append({
                        "role": "user",
                        "content": "You haven't taken any action yet. Use a tool to begin the task."
                    })
                    if self.settings.verbose:
                        console.print("    [yellow]⚠ Premature finish rejected. Forcing tool usage.[/yellow]")
                    continue
                    
                summary = response.get("content", "Task completed.")
                if self.settings.verbose:
                    console.print(f"    [green]✓ Done:[/green] {summary}")
                trace.final_answer = summary
                break

        if not trace.final_answer:
            trace.final_answer = "MAX_STEPS reached"
        trace.success = any(t.result.success for t in trace.step_traces) if getattr(trace, 'step_traces', []) else False

    async def _trim_messages(self, messages: list[dict], max_chars: int = 40000) -> list[dict]:
        """Ensures the message history doesn't exceed context limits by summarizing middle messages."""
        import json
        total_len = sum(len(json.dumps(m)) for m in messages)
        if total_len <= max_chars:
            return messages
            
        system_msgs = messages[:2]
        recent_msgs = messages[-6:] if len(messages) > 8 else []
        middle_msgs = messages[2:-6] if len(messages) > 8 else []
        
        if not middle_msgs:
            return messages
            
        try:
            middle_text = "\\n".join(json.dumps(m)[:500] for m in middle_msgs)
            summary_prompt = f"Summarize the following old tool execution history compactly:\\n{middle_text}"
            summary = await self._provider.complete(user=summary_prompt, system="You are a summarizer. Keep it extremely brief.", max_tokens=256)
            summary_msg = {"role": "system", "content": f"[HISTORY SUMMARY]\\n{summary}"}
            return system_msgs + [summary_msg] + recent_msgs
        except Exception:
            return system_msgs + [{"role": "system", "content": "[HISTORY TRUNCATED]"}] + recent_msgs

    # ── Plan Execution ───────────────────────────────────────────────────────

    async def _execute_plan_and_execute(self, task: str, plan: ExecutionPlan, trace: ExecutionTrace, context: str):
        """Sequential plan execution with ReAct per step."""
        if not self.settings.execution_enabled:
            trace.final_answer = (
                f"[Execution disabled] Plan has {len(plan.steps)} steps:\n" +
                "\n".join(f"  {i+1}. {s.tool}: {s.description}" for i, s in enumerate(plan.steps))
            )
            trace.success = True
            return

        current_plan = plan
        step_idx = 0
        replan_count = 0
        MAX_REPLANS = 3
        
        while step_idx < len(current_plan.steps):
            step = current_plan.steps[step_idx]
            
            # Create a fresh trace for the sub-step to keep step logic isolated
            sub_trace = ExecutionTrace(
                session_id=trace.session_id, 
                task=f"Step {step.step_number}: {step.description}",
                intent=getattr(trace, "intent", Intent.GENERAL_QA),
                classification=getattr(trace, "classification", None),
                plan=None
            )
            
            # Execute react loop for this step only
            await self._execute_react(f"{step.tool}: {step.description}\nArgs: {step.parameters}", ExecutionPlan(task=task, steps=[step], reasoning=""), sub_trace, context)
            
            trace.step_traces.extend(sub_trace.step_traces)
            
            if not sub_trace.success:
                if self.settings.verbose:
                    from rich.console import Console
                    console = Console()
                    console.print(f"    [yellow]⚠ Step {step.step_number} failed. Re-planning...[/yellow]")
                
                if replan_count >= MAX_REPLANS:
                    if self.settings.verbose:
                        console.print("    [red]✗ Max replans reached. Aborting plan execution.[/red]")
                    break
                
                # Context injection for replan
                replan_ctx = context + f"\n\n[PREVIOUS PLAN FAILURE]\nStep {step.step_number} failed. Sub-trace final answer: {sub_trace.final_answer}\nUpdate the plan to recover."
                new_plan, _ = await self._committee._planner.plan(task, context=replan_ctx)
                if new_plan and new_plan.steps:
                    current_plan = new_plan
                    step_idx = 0
                    replan_count += 1
                    continue
                else:
                    break
            
            step_idx += 1
            
        await self._reflect(trace)
        
    async def _reflect(self, trace: ExecutionTrace):
        """Reflection pass to verify if the task was completed successfully."""
        from sumospace.schemas import dereference_schema
        system = "You are a Reflection Agent. Review the execution trace and determine if the task was completed successfully. If not, trigger a retry."
        
        schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "reason": {"type": "string"},
                "retry": {"type": "boolean"}
            },
            "required": ["success", "reason", "retry"],
            "additionalProperties": False
        }
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": "submit_reflection",
                "description": "Submit reflection results",
                "parameters": schema
            }
        }
        
        trace_summary = "\\n".join(f"Step {t.step_number} ({t.tool}): {'Success' if t.result.success else 'Failed'} - {t.result.output[:100]}" for t in trace.step_traces)
        prompt = f"Task: {trace.task}\n\nTrace Summary:\n{trace_summary}"
        
        try:
            response = await self._provider.complete_with_tools(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                tools=[tool_schema]
            )
            
            if response.get("type") == "tool_calls" and response.get("tool_calls"):
                tc = response["tool_calls"][0]
                import json
                args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                
                trace.success = args.get("success", False)
                trace.final_answer = f"Reflection: {args.get('reason', '')}"
                
                if args.get("retry") and not trace.success:
                    # In a real scenario we'd loop the outer kernel, but for this sprint
                    # we just note it.
                    trace.final_answer += " (Retry needed)"
        except Exception as e:
            if getattr(self.settings, "verbose", False):
                print(f"Reflection failed: {e}")


    # ── Synthesis ────────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        task: str,
        trace: ExecutionTrace,
        context: str,
    ) -> AsyncIterator[str]:
        """Generate a natural language response fulfilling the user's task."""
        outputs = "\n\n".join([
            f"Step {t.step_number} ({t.tool}): {t.result.output[:4000]}"
            for t in trace.step_traces
            if t.result.output
        ])

        system = (
            "You are a helpful assistant fulfilling the user's task.\n"
            "You have executed a set of tools to gather information or perform actions.\n"
            "Given the user's original task and the outputs from the tools you ran, "
            "provide a comprehensive and direct answer to the task.\n"
            "If the task asks for an explanation, explain it fully using the tool outputs.\n"
            "If the task was an action (e.g., editing files), summarize what changes were made."
        )
        user = f"Task: {task}\n\nTool outputs:\n{outputs[:16000]}"
        if context:
            user += f"\n\nContext used:\n{context[:2000]}"

        try:
            async with self.telemetry.async_span("sumospace.synthesise"):
                async for chunk in self._provider.stream(user=user, system=system, temperature=0.1):
                    yield chunk
        except Exception:
            yield outputs or "Task completed."

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_full_context(
        self, 
        task: str, 
        rag_context: str = "", 
        web_context: str = "", 
        memory_str: str = "",
        max_tokens: int = 4096
    ) -> str:
        """
        Builds a comprehensive context string for the LLM, 
        applying priority-based truncation to fit within max_tokens.
        """
        from sumospace.utils.tokens import truncate_by_tokens
        
        # 1. Budgeting (fixed ratios)
        # Tools & Session: ~10% (Static)
        # Task: 15%
        # Memory: 20%
        # Web: 15%
        # RAG: 40% (Lowest priority, truncated first)
        
        mem_budget = int(max_tokens * 0.20)
        web_budget = int(max_tokens * 0.15)
        rag_budget = int(max_tokens * 0.40)
        
        # Tools & Session Info
        tools_list = self._tools.list_tools()
        tools_str = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
        session_info = f"user_id: {self.settings.user_id}\nsession_id: {self.settings.session_id}"
        
        # Truncation logic (Priority: Task > Memory > Web > RAG)
        truncated_rag = truncate_by_tokens(rag_context, rag_budget) if rag_context else ""
        truncated_web = truncate_by_tokens(web_context, web_budget) if web_context else ""
        truncated_mem = truncate_by_tokens(memory_str, mem_budget) if memory_str else ""
        
        parts = [
            f"=== AVAILABLE TOOLS ===\n{tools_str}",
            f"=== SESSION CONTEXT ===\n{session_info}",
            f"=== TASK ===\n{task}",
        ]
        
        # Fast Workspace Snapshot (to prevent static planner hallucinations)
        import os
        from pathlib import Path
        workspace_snapshot = ""
        ws_path = Path(self.settings.workspace)
        if ws_path.exists() and ws_path.is_dir():
            try:
                tree = []
                py_contents = []
                for root, dirs, files in os.walk(ws_path):
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ["__pycache__", "node_modules", "venv"]]
                    rel_root = Path(root).relative_to(ws_path)
                    for f in files:
                        if f.startswith('.'): continue
                        rel_path = rel_root / f if str(rel_root) != "." else Path(f)
                        tree.append(str(rel_path))
                        if f.endswith('.py') and len(py_contents) < 8:
                            file_path = Path(root) / f
                            if file_path.stat().st_size < 15000:
                                py_contents.append(f"--- {rel_path} ---\n{file_path.read_text(errors='replace')}")
                
                workspace_snapshot = "Files in workspace:\n" + "\n".join(tree)
                if py_contents:
                    workspace_snapshot += "\n\nFile contents (preview):\n" + "\n\n".join(py_contents)
            except Exception:
                pass
                
        if workspace_snapshot:
            parts.append(f"=== WORKSPACE SNAPSHOT ===\n{truncate_by_tokens(workspace_snapshot, 2048)}")
        
        if truncated_mem:
            parts.append(f"=== RECENT MEMORY ===\n{truncated_mem}")
        if truncated_web:
            parts.append(f"=== WEB SEARCH RESULTS ===\n{truncated_web}")
        if truncated_rag:
            parts.append(f"=== CODEBASE CONTEXT ===\n{truncated_rag}")
            
        return "\n\n".join(parts)

    def _format_dry_run(self, verdict: CommitteeVerdict) -> str:
        lines = [
            f"[DRY RUN] Task: {verdict.plan.task}",
            f"Approved: {verdict.approved}",
            f"Steps planned: {len(verdict.plan.steps)}",
            "",
        ]
        for step in verdict.plan.steps:
            lines.append(
                f"  Step {step.step_number}: [{step.tool}] {step.description}"
            )
            if step.parameters:
                import json
                lines.append(f"    Params: {json.dumps(step.parameters)[:200]}")
        return "\n".join(lines)

    def _auto_load_hooks(self) -> None:
        """Auto-load hooks from workspace .sumo_hooks.py if enabled."""
        import importlib.util
        from pathlib import Path as _Path

        # Explicit module path
        if self.settings.hooks_module:
            self._load_hooks_from_path(self.settings.hooks_module)
            return

        # Auto-discovery from workspace (gated by setting)
        if self.settings.auto_load_hooks:
            hooks_file = _Path(self.settings.workspace) / ".sumo_hooks.py"
            if hooks_file.is_file():
                self._load_hooks_from_path(str(hooks_file))

    def _load_hooks_from_path(self, path: str) -> None:
        """Import a Python file and register any decorated hooks."""
        import importlib.util
        from pathlib import Path as _Path

        p = _Path(path)
        if not p.is_file():
            console.print(f"[yellow]Hooks file not found: {path}[/yellow]")
            return

        try:
            spec = importlib.util.spec_from_file_location("sumo_hooks", str(p))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Inject hooks registry so decorators work
                module.hooks = self.hooks  # type: ignore
                spec.loader.exec_module(module)
                if self.settings.verbose:
                    console.print(f"[dim]Loaded hooks from {path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to load hooks from {path}: {e}[/yellow]")

    # ── Convenience methods ──────────────────────────────────────────────────

    async def ingest(
        self,
        path: str,
        recursive: bool = True,
    ) -> int:
        """Ingest a file or directory into the RAG knowledge base."""
        if not self._initialized:
            await self.boot()
        from pathlib import Path
        p = Path(path)
        if p.is_dir():
            results = await self._ingestor.ingest_directory(path)
            return sum(r.chunks_created for r in results)
        else:
            result = await self._ingestor.ingest_file(path)
            return result.chunks_created

    async def ingest_media(self, path: str, force: bool = False) -> list[Any]:
        """
        Ingest text, images, audio, or video.
        Requires settings.media_enabled = True.
        """
        if not self.settings.media_enabled:
            raise ValueError("Media features are disabled. Set media_enabled=True in settings.")
        if not self._initialized:
            await self.boot()
        
        from sumospace.media_ingest import MultimodalIngestor
        ingestor = MultimodalIngestor(self.settings)
        return ingestor.ingest_path(path, force=force)

    async def search_media(self, query: str, top_k: int = 3) -> list[Any]:
        """
        Search across all modalities. Query can be text, or path to image/audio/video.
        Requires settings.media_enabled = True.
        """
        if not self.settings.media_enabled:
            raise ValueError("Media features are disabled. Set media_enabled=True in settings.")
        if not self._initialized:
            await self.boot()
            
        from sumospace.media_search import MultimodalSearchEngine
        engine = MultimodalSearchEngine(self.settings)
        return engine.search(query, top_k=top_k)

    async def recall(self, query: str, top_k: int = 5):
        """Direct semantic recall from memory."""
        if not self._initialized:
            await self.boot()
        return await self._memory.recall(query, top_k=top_k)

    async def chat(self, message: str, session_id: str | None = None) -> str:
        """Simple conversational turn (no tool execution)."""
        if not self._initialized:
            await self.boot()
        await self._memory.add("user", message)
        recent = self._memory.recent(10)
        history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent[:-1])
        system = "You are Sumo, a helpful AI assistant."
        user = f"{history}\n\nUSER: {message}" if history else message
        response = await self._provider.complete(user=user, system=system, temperature=0.1)
        await self._memory.add("assistant", response)
        return response
