# sumospace/benchmarks/report.py
"""
Benchmark Report Generator
===========================
Outputs benchmark results as JSON and Markdown.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sumospace.benchmarks.runner import BenchmarkResult, TaskResult


class BenchmarkReporter:
    """Generate JSON and Markdown reports from benchmark results."""

    def __init__(self, results: list[BenchmarkResult]):
        self._results = results

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# SumoSpace Benchmark Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Summary table
        lines += [
            "## Summary",
            "",
            "| Committee Mode | Success Rate | Avg Duration | Tool Failures |",
            "|----------------|-------------|--------------|---------------|",
        ]
        for r in self._results:
            lines.append(
                f"| `{r.committee_mode}` | "
                f"{r.success_rate * 100:.0f}% ({sum(1 for t in r.task_results if t.passed)}/{len(r.task_results)}) | "
                f"{r.avg_duration_s:.1f}s | "
                f"{r.total_tool_failures} |"
            )

        lines += [""]

        # Per-task breakdown per mode
        for result in self._results:
            lines += [
                f"## Mode: `{result.committee_mode}`",
                "",
                f"- **Run ID**: `{result.run_id}`",
                f"- **Success Rate**: {result.success_rate * 100:.0f}%",
                f"- **Total Duration**: {sum(t.duration_s for t in result.task_results):.1f}s",
                "",
                "| Task | Passed | Duration | Retries | Tool Calls | Reason |",
                "|------|--------|----------|---------|------------|--------|",
            ]
            for task in result.task_results:
                status = "✅" if task.passed else "❌"
                lines.append(
                    f"| {task.task_name} | {status} | "
                    f"{task.duration_s:.1f}s | {task.retries} | "
                    f"{task.tool_calls} | {task.validation_reason} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str = "."):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = out / f"benchmark_{ts}.json"
        md_path = out / f"benchmark_{ts}.md"

        json_path.write_text(self.to_json(), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        return str(json_path), str(md_path)

    def _to_dict(self) -> dict:
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": [
                {
                    "run_id": r.run_id,
                    "committee_mode": r.committee_mode,
                    "success_rate": round(r.success_rate, 3),
                    "avg_duration_s": round(r.avg_duration_s, 2),
                    "total_tool_failures": r.total_tool_failures,
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "task_name": t.task_name,
                            "passed": t.passed,
                            "validation_reason": t.validation_reason,
                            "duration_s": t.duration_s,
                            "retries": t.retries,
                            "tool_calls": t.tool_calls,
                            "tool_failures": t.tool_failures,
                            "rollback_triggered": t.rollback_triggered,
                            "error": t.error,
                        }
                        for t in r.task_results
                    ],
                }
                for r in self._results
            ],
        }
