# sumospace/benchmarks/__init__.py
"""
SumoSpace Benchmark Framework
==============================
Reproducible evaluation of agent quality against a bundled fixture codebase.

CLI:
    sumo benchmark run
    sumo benchmark compare
    sumo benchmark report
"""
from sumospace.benchmarks.tasks import BenchmarkTask, TASK_REGISTRY
from sumospace.benchmarks.runner import BenchmarkRunner, BenchmarkResult
from sumospace.benchmarks.report import BenchmarkReporter

__all__ = [
    "BenchmarkTask", "TASK_REGISTRY",
    "BenchmarkRunner", "BenchmarkResult",
    "BenchmarkReporter",
]
