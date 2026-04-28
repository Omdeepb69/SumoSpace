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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from sumospace.classifier import ClassificationResult, Intent, SumoClassifier
from sumospace.committee import Committee, CommitteeVerdict, ExecutionPlan
from sumospace.exceptions import (
    ConsensusFailedError,
    ExecutionHaltedError,
    KernelBootError,
)
from sumospace.ingest import UniversalIngestor
from sumospace.memory import MemoryManager
from sumospace.providers import ProviderRouter
from sumospace.rag import RAGPipeline
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

    # ── Paths ─────────────────────────────────────────────────────────────────
    workspace: str = "."
    chroma_path: str = ".sumo_db"     # Direct override — bypasses ScopeManager

    # ── Scope & isolation ─────────────────────────────────────────────────────
    scope_level: str = "user"         # user | session | project
    user_id: str = ""                 # Already-validated user identifier
    session_id: str = ""              # Session identifier (for session-level scope)
    project_id: str = ""              # Project identifier (for project-level scope)
    chroma_base: str = ".sumo_db"     # Base directory for scoped DB paths
    max_chunks_per_scope: int | None = None   # Quota guard per scope (None = unlimited)


# ─── Execution Trace ──────────────────────────────────────────────────────────

@dataclass
class StepTrace:
    step_number: int
    tool: str
    description: str
    result: ToolResult
    duration_ms: float


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

    def __init__(self, config: KernelConfig | None = None):
        self.config = config or KernelConfig()
        self._provider: ProviderRouter | None = None
        self._classifier: SumoClassifier | None = None
        self._committee: Committee | None = None
        self._tools: ToolRegistry | None = None
        self._memory: MemoryManager | None = None
        self._ingestor: UniversalIngestor | None = None
        self._rag: RAGPipeline | None = None
        self._initialized = False

    async def boot(self):
        """Initialise all subsystems. Called automatically by async context manager."""
        if self._initialized:
            return

        cfg = self.config
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
            )
            await self._provider.initialize()

            # 2. Tool registry
            self._tools = ToolRegistry(workspace=cfg.workspace)

            # 3. Scope resolution
            #    If user_id is set, build a ScopeManager and resolve paths.
            #    Otherwise fall back to raw chroma_path for backward compat.
            scope_mgr = None
            resolved_chroma = cfg.chroma_path
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
            self._memory = MemoryManager(
                chroma_path=resolved_chroma,
                embedding_provider=cfg.embedding_provider,
                scope_manager=scope_mgr,
                user_id=cfg.user_id,
                session_id=cfg.session_id,
                project_id=cfg.project_id,
            )
            await self._memory.initialize()

            # 5. Ingestor + RAG
            self._ingestor = UniversalIngestor(
                chroma_path=resolved_chroma,
                embedding_provider=cfg.embedding_provider,
                embedding_model=cfg.embedding_model,
                max_chunks=cfg.max_chunks_per_scope,
            )
            await self._ingestor.initialize()

            self._rag = RAGPipeline(ingestor=self._ingestor)
            await self._rag.initialize()

            # 5. Classifier
            self._classifier = SumoClassifier(provider=self._provider)
            await self._classifier.initialize()

            # 6. Committee
            self._committee = Committee(
                provider=self._provider,
                require_consensus=cfg.require_consensus,
            )

            self._initialized = True

            if cfg.verbose:
                console.print("[bold green]✓ Kernel ready[/bold green]")

        except Exception as e:
            raise KernelBootError(f"Kernel boot failed: {e}") from e

    async def shutdown(self):
        """Graceful shutdown."""
        self._initialized = False
        if cfg := self.config:
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
        Execute a task end-to-end.

        Args:
            task:       Natural language task description.
            session_id: Optional session identifier for memory scoping.

        Returns:
            ExecutionTrace with full audit trail.
        """
        if not self._initialized:
            await self.boot()

        session_id = session_id or uuid.uuid4().hex[:12]
        start = time.monotonic()

        trace = ExecutionTrace(
            task=task,
            session_id=session_id,
            intent=Intent.GENERAL_QA,
            classification=None,
            plan=None,
        )

        try:
            # Step 1: Classify
            if self.config.verbose:
                console.print(f"\n[bold]Task:[/bold] {task}")

            recent_ctx = {
                "recent_messages": [m["content"] for m in self._memory.recent(5)],
            }
            classification = await self._classifier.classify(task, context=recent_ctx)
            trace.intent = classification.intent
            trace.classification = classification

            if self.config.verbose:
                console.print(
                    f"[dim]Intent: [cyan]{classification.intent.value}[/cyan] "
                    f"({classification.confidence:.0%}) — {classification.reasoning}[/dim]"
                )

            # Step 2: RAG retrieval (if needed)
            rag_context = ""
            if classification.needs_retrieval:
                try:
                    rag_result = await self._rag.retrieve(task)
                    if rag_result.chunks:
                        rag_context = rag_result.context
                        trace.rag_context = rag_context
                        if self.config.verbose:
                            console.print(
                                f"[dim]Retrieved {len(rag_result.chunks)} chunks "
                                f"(reranked: {rag_result.used_reranker})[/dim]"
                            )
                except Exception as e:
                    if self.config.verbose:
                        console.print(f"[yellow]RAG skipped: {e}[/yellow]")

            # Step 3: Web search (if needed)
            web_context = ""
            if classification.needs_web:
                web_result = await self._tools.execute("web_search", query=task)
                if web_result.success:
                    web_context = web_result.output

            # Step 4: Build full context
            context_parts = []
            if rag_context:
                context_parts.append(f"=== CODEBASE CONTEXT ===\n{rag_context}")
            if web_context:
                context_parts.append(f"=== WEB SEARCH RESULTS ===\n{web_context}")
            if self._memory.recent(5):
                mem_str = self._memory.context_string(5)
                context_parts.append(f"=== RECENT MEMORY ===\n{mem_str}")
            full_context = "\n\n".join(context_parts)

            # Step 5: Committee deliberation
            if self.config.verbose:
                console.print("[dim]Committee deliberating...[/dim]")

            verdict = await self._committee.deliberate(task, context=full_context)
            trace.plan = verdict.plan

            if not verdict.approved:
                raise ConsensusFailedError(f"Committee rejected plan: {verdict.rejection_reason}")

            if self.config.verbose:
                console.print(
                    f"[green]✓ Plan approved[/green] — "
                    f"{len(verdict.plan.steps)} steps, "
                    f"~{verdict.plan.estimated_duration_s:.0f}s estimated"
                )

            # Step 6: Execute (skip if dry_run)
            if self.config.dry_run:
                trace.final_answer = self._format_dry_run(verdict)
                trace.success = True
            else:
                await self._execute_plan(verdict.plan, trace)

            # Step 7: Synthesise final answer
            if not trace.final_answer:
                trace.final_answer = await self._synthesise(task, trace, full_context)

            # Step 8: Persist to memory
            await self._memory.add("user", task)
            await self._memory.add("assistant", trace.final_answer)

        except ConsensusFailedError as e:
            trace.error = str(e)
            trace.success = False
            trace.final_answer = f"Task halted: {e}"
            if self.config.verbose:
                console.print(f"[red]✗ {e}[/red]")

        except ExecutionHaltedError as e:
            trace.error = str(e)
            trace.success = False
            trace.final_answer = f"Execution halted at critical step: {e}"
            if self.config.verbose:
                console.print(f"[red]✗ {e}[/red]")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            trace.final_answer = f"Unexpected error: {e}"
            if self.config.verbose:
                console.print_exception()

        trace.duration_ms = (time.monotonic() - start) * 1000

        if self.config.verbose:
            status = "[green]✓ Done[/green]" if trace.success else "[red]✗ Failed[/red]"
            console.print(
                f"{status} in {trace.duration_ms:.0f}ms — "
                f"{len(trace.step_traces)} steps executed"
            )

        return trace

    # ── Plan Execution ───────────────────────────────────────────────────────

    async def _execute_plan(self, plan: ExecutionPlan, trace: ExecutionTrace):
        """Execute each step in the approved plan sequentially."""
        for step in plan.steps:
            if self.config.verbose:
                console.print(
                    f"  [cyan][{step.step_number}/{len(plan.steps)}][/cyan] "
                    f"{step.tool}: {step.description}"
                )

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

            if self.config.verbose:
                if result.success:
                    preview = result.output[:120].replace("\n", " ")
                    console.print(f"    [green]✓[/green] {preview}{'...' if len(result.output) > 120 else ''}")
                else:
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
    ) -> str:
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
            return await self._provider.complete(
                user=user, system=system, temperature=0.2, max_tokens=1024
            )
        except Exception:
            return outputs or "Task completed."

    # ── Helpers ──────────────────────────────────────────────────────────────

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
        response = await self._provider.complete(user=user, system=system, temperature=0.3)
        await self._memory.add("assistant", response)
        return response
