# sumospace/hooks.py

"""
Kernel Lifecycle Hooks for SumoSpace.
======================================
Provides a decorator-friendly ``HookRegistry`` that fires async (or sync)
callbacks at well-defined points in the kernel lifecycle.

Usage::

    from sumospace.hooks import HookRegistry

    hooks = HookRegistry()

    @hooks.on("on_task_start")
    async def log_start(task, session_id):
        print(f"Starting: {task}")

    kernel = SumoKernel(settings=settings, hooks=hooks)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console

console = Console()

# ── Allowed hook events ───────────────────────────────────────────────────────

HOOK_EVENTS: set[str] = {
    "on_kernel_boot",       # (kernel: SumoKernel)
    "on_kernel_shutdown",   # (kernel: SumoKernel)
    "on_task_start",        # (task: str, session_id: str)
    "on_plan_approved",     # (plan: ExecutionPlan, verdict: CommitteeVerdict)
    "on_plan_rejected",     # (reason: str, verdict: CommitteeVerdict)
    "on_step_start",        # (step: ExecutionStep)
    "on_step_complete",     # (trace: StepTrace)
    "on_step_failed",       # (step: ExecutionStep, error: str)
    "on_task_complete",     # (trace: ExecutionTrace)
    "on_task_failed",       # (trace: ExecutionTrace, error: str)
}


class HookRegistry:
    """
    Registry for kernel lifecycle hooks.

    Hooks are async-native but sync callbacks are supported transparently
    via ``asyncio.run_in_executor``. Hook exceptions are logged but never
    propagated — hooks must never crash the kernel.
    """

    def __init__(self, verbose: bool = True):
        self._hooks: dict[str, list[Callable]] = {}
        self._verbose = verbose

    # ── Registration ──────────────────────────────────────────────────────────

    def on(self, event: str) -> Callable:
        """
        Decorator for registering hooks::

            @hooks.on("on_task_start")
            async def my_hook(task, session_id):
                ...
        """
        if event not in HOOK_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid: {sorted(HOOK_EVENTS)}"
            )

        def decorator(fn: Callable) -> Callable:
            self._hooks.setdefault(event, []).append(fn)
            return fn

        return decorator

    def register(self, event: str, fn: Callable) -> None:
        """Imperative registration for programmatic use."""
        if event not in HOOK_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid: {sorted(HOOK_EVENTS)}"
            )
        self._hooks.setdefault(event, []).append(fn)

    def clear(self, event: str | None = None) -> None:
        """Clear all hooks, or hooks for a specific event."""
        if event:
            self._hooks.pop(event, None)
        else:
            self._hooks.clear()

    # ── Triggering ────────────────────────────────────────────────────────────

    async def trigger(self, event: str, *args: Any, **kwargs: Any) -> None:
        """
        Fire all hooks registered for ``event``.

        - Async hooks are awaited directly.
        - Sync hooks run via ``run_in_executor`` to avoid blocking.
        - All exceptions are caught and logged — hooks never crash the kernel.
        """
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: hook(*args, **kwargs)
                    )
            except Exception as e:
                if self._verbose:
                    name = getattr(hook, "__name__", repr(hook))
                    console.print(
                        f"[yellow]Hook '{name}' on '{event}' failed: {e}[/yellow]"
                    )

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def registered_events(self) -> list[str]:
        """Return events that have at least one hook registered."""
        return sorted(k for k, v in self._hooks.items() if v)

    def count(self, event: str | None = None) -> int:
        """Count hooks for a specific event, or all hooks."""
        if event:
            return len(self._hooks.get(event, []))
        return sum(len(v) for v in self._hooks.values())
