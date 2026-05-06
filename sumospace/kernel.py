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
    duration_ms: float = 0.0
    rag_context: str = ""

    @property
    def tool_outputs(self) -> list[str]:
        return [t.result.output for t in self.step_traces]

    @property
    def failed_steps(self) -> list[StepTrace]:
        return [t for t in self.step_traces if not t.result.success]


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
            self._tools = ToolRegistry(workspace=cfg.workspace)

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
                self._ingestor = UniversalIngestor(
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
                    web_result = await self._tools.execute("web_search", query=task)
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
            else:
                async with self.telemetry.async_span("sumospace.execute", attributes={"steps": len(verdict.plan.steps)}):
                    await self._execute_plan(verdict.plan, trace)

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
            trace.final_answer = f"Task halted: {e}"
            if self.settings.verbose:
                console.print(f"[red]✗ {e}[/red]")

        except ExecutionHaltedError as e:
            trace.error = str(e)
            trace.success = False
            trace.final_answer = f"Execution halted at critical step: {e}"
            if self.settings.verbose:
                console.print(f"[red]✗ {e}[/red]")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
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
                    web_result = await self._tools.execute("web_search", query=task)
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
                    result = await self._tools.execute(step.tool, **step.parameters)
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

    # ── Plan Execution ───────────────────────────────────────────────────────

    async def _execute_plan(self, plan: ExecutionPlan, trace: ExecutionTrace):
        """Execute each step in the approved plan sequentially."""
        if not self.settings.execution_enabled:
            trace.final_answer = (
                f"[Execution disabled] Plan has {len(plan.steps)} steps:\n" +
                "\n".join(f"  {i+1}. {s.tool}: {s.description}" for i, s in enumerate(plan.steps))
            )
            trace.success = True
            return
            
        for step in plan.steps:
            if self.settings.verbose:
                console.print(
                    f"  [cyan][{step.step_number}/{len(plan.steps)}][/cyan] "
                    f"{step.tool}: {step.description}"
                )

            await self.hooks.trigger("on_step_start", step)

            step_start = time.monotonic()
            async with self.telemetry.async_span(f"sumospace.tool.{step.tool}", attributes={"description": step.description}):
                result = await self._tools.execute(step.tool, **step.parameters)
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
                if self.settings.verbose:
                    preview = result.output[:120].replace("\n", " ")
                    console.print(f"    [green]✓[/green] {preview}{'...' if len(result.output) > 120 else ''}")
            else:
                await self.hooks.trigger("on_step_failed", step, result.error)
                if self.settings.verbose:
                    console.print(f"    [red]✗ {result.error}[/red]")

            if not result.success and step.critical:
                raise ExecutionHaltedError(
                    f"Critical step {step.step_number} ({step.tool}) failed: {result.error}"
                )

        trace.success = all(t.result.success or not t.result.error for t in trace.step_traces)

    # ── Synthesis ────────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        task: str,
        trace: ExecutionTrace,
        context: str,
    ) -> AsyncIterator[str]:
        """Generate a natural language summary of what was done."""
        outputs = "\n\n".join([
            f"Step {t.step_number} ({t.tool}): {t.result.output[:500]}"
            for t in trace.step_traces
            if t.result.output
        ])

        system = (
            "You are a helpful assistant summarising completed tasks.\n"
            "Given the task and tool outputs, write a clear, concise summary of what was done.\n"
            "Be specific. Mention files changed, commands run, or answers found."
        )
        user = f"Task: {task}\n\nTool outputs:\n{outputs[:3000]}"
        if context:
            user += f"\n\nContext used:\n{context[:1000]}"

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
        
        from sumospace.media_ingest import MediaIngestor
        ingestor = MediaIngestor(self.settings)
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
            
        from sumospace.media_search import MediaSearchEngine
        engine = MediaSearchEngine(self.settings)
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
