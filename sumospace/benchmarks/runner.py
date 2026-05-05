# sumospace/benchmarks/runner.py
"""
Benchmark Runner
================
Executes benchmark tasks through SumoKernel and records metrics.
Supports committee mode comparison.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sumospace.benchmarks.tasks import BenchmarkTask, TASK_REGISTRY, FIXTURES_DIR


@dataclass
class TaskResult:
    task_id: str
    task_name: str
    committee_mode: str
    passed: bool
    validation_reason: str
    duration_s: float
    retries: int
    tool_calls: int
    tool_failures: int
    rollback_triggered: bool
    error: str = ""
    output_excerpt: str = ""


@dataclass
class BenchmarkResult:
    run_id: str
    timestamp: float
    committee_mode: str
    workspace: str
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.passed) / len(self.task_results)

    @property
    def avg_duration_s(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.duration_s for r in self.task_results) / len(self.task_results)

    @property
    def total_tool_failures(self) -> int:
        return sum(r.tool_failures for r in self.task_results)


class BenchmarkRunner:
    """
    Runs benchmark tasks through SumoKernel.

    Args:
        settings:        SumoSettings instance. committee_mode is overridden per run.
        workspace:       Directory to run tasks against. Defaults to bundled fixtures.
        task_ids:        Subset of task IDs to run. None = all tasks.
        committee_modes: List of modes to compare. E.g. ["full", "plan_only", "disabled"].
    """

    VALID_MODES = ["full", "plan_only", "critique_only", "disabled"]

    def __init__(
        self,
        settings,
        workspace: str | None = None,
        task_ids: list[str] | None = None,
        committee_modes: list[str] | None = None,
    ):
        self._settings = settings
        self._workspace = workspace or str(FIXTURES_DIR)
        self._task_ids = task_ids
        self._modes = committee_modes or ["full"]

    def _get_tasks(self) -> list[BenchmarkTask]:
        if not self._task_ids:
            return TASK_REGISTRY
        return [t for t in TASK_REGISTRY if t.id in self._task_ids]

    async def run(self) -> list[BenchmarkResult]:
        """Run all tasks across all committee modes."""
        results = []
        for mode in self._modes:
            result = await self._run_mode(mode)
            results.append(result)
        return results

    async def _run_mode(self, committee_mode: str) -> BenchmarkResult:
        import hashlib

        run_id = hashlib.sha256(
            f"{committee_mode}{time.time()}".encode()
        ).hexdigest()[:12]

        result = BenchmarkResult(
            run_id=run_id,
            timestamp=time.time(),
            committee_mode=committee_mode,
            workspace=self._workspace,
        )

        tasks = self._get_tasks()
        for task in tasks:
            task_result = await self._run_task(task, committee_mode, run_id)
            result.task_results.append(task_result)

        return result

    async def _run_task(
        self, task: BenchmarkTask, committee_mode: str, run_id: str
    ) -> TaskResult:
        """Run a single task in a temp copy of the workspace."""
        from sumospace.settings import SumoSettings
        from sumospace.kernel import SumoKernel

        # Work in a temp copy so fixture files are never modified
        with tempfile.TemporaryDirectory() as tmp_ws:
            shutil.copytree(self._workspace, tmp_ws, dirs_exist_ok=True)

            # Build settings for this mode
            mode_settings = SumoSettings(
                **{
                    **self._settings.model_dump(),
                    "workspace": tmp_ws,
                    "committee_enabled": committee_mode != "disabled",
                    "committee_mode": committee_mode if committee_mode != "disabled" else "full",
                    "dry_run": False,
                    "verbose": False,
                }
            )

            start = time.monotonic()
            tool_calls = 0
            tool_failures = 0
            retries = 0
            rollback = False
            output_text = ""
            error = ""

            try:
                async with SumoKernel(settings=mode_settings) as kernel:
                    trace = await kernel.run(
                        task=task.prompt,
                        workspace=tmp_ws,
                    )
                    output_text = str(getattr(trace, "final_output", trace))
                    # Collect metrics from trace if available
                    tool_calls = len(getattr(trace, "tool_calls", []))
                    tool_failures = sum(
                        1 for tc in getattr(trace, "tool_calls", [])
                        if not getattr(tc, "success", True)
                    )
                    retries = getattr(trace, "retry_count", 0)
                    rollback = getattr(trace, "rolled_back", False)
            except Exception as e:
                error = str(e)

            duration = time.monotonic() - start

            # Validate
            passed, reason = False, "Kernel error — no output"
            if not error and task.validator:
                try:
                    passed, reason = task.validator(output_text, tmp_ws)
                except Exception as ve:
                    reason = f"Validator error: {ve}"

            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                committee_mode=committee_mode,
                passed=passed,
                validation_reason=reason,
                duration_s=round(duration, 2),
                retries=retries,
                tool_calls=tool_calls,
                tool_failures=tool_failures,
                rollback_triggered=rollback,
                error=error,
                output_excerpt=output_text[:300] if output_text else "",
            )
