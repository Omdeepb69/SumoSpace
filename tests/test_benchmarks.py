# tests/test_benchmarks.py
"""Tests for benchmark framework."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sumospace.benchmarks.tasks import TASK_REGISTRY, _validate_docstrings, _validate_async
from sumospace.benchmarks.report import BenchmarkReporter
from sumospace.benchmarks.runner import BenchmarkResult, TaskResult


# ── Task definitions ───────────────────────────────────────────────────────

def test_task_registry_has_five_tasks():
    assert len(TASK_REGISTRY) == 5


def test_all_tasks_have_validators():
    for task in TASK_REGISTRY:
        assert task.validator is not None, f"Task '{task.id}' has no validator"


def test_all_tasks_have_prompts():
    for task in TASK_REGISTRY:
        assert len(task.prompt) > 20, f"Task '{task.id}' has a suspiciously short prompt"


# ── Validators ─────────────────────────────────────────────────────────────

def test_validate_docstrings_pass(tmp_path):
    (tmp_path / "a.py").write_text(
        'def foo():\n    """Does foo."""\n    pass\n'
        'def bar():\n    """Does bar."""\n    pass\n'
        'def baz():\n    """Does baz."""\n    pass\n'
    )
    passed, reason = _validate_docstrings("", str(tmp_path))
    assert passed, reason


def test_validate_docstrings_fail(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    passed, reason = _validate_docstrings("", str(tmp_path))
    assert not passed


def test_validate_async_pass(tmp_path):
    (tmp_path / "database.py").write_text(
        "async def get(): pass\nasync def create(): pass\nasync def update(): pass\n"
    )
    passed, reason = _validate_async("", str(tmp_path))
    assert passed, reason


def test_validate_async_fail(tmp_path):
    (tmp_path / "database.py").write_text("def get(): pass\n")
    passed, reason = _validate_async("", str(tmp_path))
    assert not passed


# ── Reporter ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_results():
    task_result = TaskResult(
        task_id="add_docstrings",
        task_name="Add Missing Docstrings",
        committee_mode="full",
        passed=True,
        validation_reason="3 functions with docstrings",
        duration_s=5.2,
        retries=0,
        tool_calls=4,
        tool_failures=0,
        rollback_triggered=False,
    )
    return [BenchmarkResult(
        run_id="abc123",
        timestamp=0,
        committee_mode="full",
        workspace="/tmp/fixtures",
        task_results=[task_result],
    )]


def test_reporter_to_markdown(sample_results):
    reporter = BenchmarkReporter(sample_results)
    md = reporter.to_markdown()
    assert "# SumoSpace Benchmark Report" in md
    assert "full" in md
    assert "Add Missing Docstrings" in md
    assert "✅" in md


def test_reporter_to_json(sample_results):
    import json
    reporter = BenchmarkReporter(sample_results)
    data = json.loads(reporter.to_json())
    assert "runs" in data
    assert data["runs"][0]["committee_mode"] == "full"
    assert data["runs"][0]["success_rate"] == 1.0


def test_reporter_save(sample_results, tmp_path):
    reporter = BenchmarkReporter(sample_results)
    json_path, md_path = reporter.save(str(tmp_path))
    assert Path(json_path).exists()
    assert Path(md_path).exists()
    assert Path(md_path).stat().st_size > 0
