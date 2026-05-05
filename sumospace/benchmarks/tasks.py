# sumospace/benchmarks/tasks.py
"""
Benchmark Task Definitions
===========================
Five reproducible tasks against the bundled sample_project fixture.
Each task has a prompt, success criteria, and validation logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_project"


@dataclass
class BenchmarkTask:
    id: str
    name: str
    description: str
    prompt: str
    workspace: str = str(FIXTURES_DIR)
    # Callable(result_text: str, workspace: str) -> (passed: bool, reason: str)
    validator: Callable[[str, str], tuple[bool, str]] | None = None
    tags: list[str] = field(default_factory=list)


def _validate_docstrings(output: str, workspace: str) -> tuple[bool, str]:
    """Check that at least 3 functions now have docstrings."""
    try:
        import ast
        py_files = list(Path(workspace).glob("*.py"))
        total_with_doc = 0
        for f in py_files:
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node):
                        total_with_doc += 1
        passed = total_with_doc >= 3
        return passed, f"{total_with_doc} functions with docstrings found"
    except Exception as e:
        return False, f"Validation error: {e}"


def _validate_async(output: str, workspace: str) -> tuple[bool, str]:
    """Check that database.py now uses async def for at least 3 functions."""
    try:
        import ast
        db_file = Path(workspace) / "database.py"
        if not db_file.exists():
            return False, "database.py not found"
        tree = ast.parse(db_file.read_text())
        async_count = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        )
        passed = async_count >= 3
        return passed, f"{async_count} async functions found in database.py"
    except Exception as e:
        return False, f"Validation error: {e}"


def _validate_dead_code(output: str, workspace: str) -> tuple[bool, str]:
    """Check that dead code markers are removed from utils.py."""
    try:
        content = (Path(workspace) / "utils.py").read_text()
        dead_markers = ["_deprecated_format", "_UNUSED_CONSTANT", "_LEGACY_SECRET"]
        remaining = [m for m in dead_markers if m in content]
        passed = len(remaining) == 0
        return passed, f"Remaining dead code: {remaining or 'none'}"
    except Exception as e:
        return False, f"Validation error: {e}"


def _validate_deps(output: str, workspace: str) -> tuple[bool, str]:
    """Check that requirements.txt versions are bumped."""
    try:
        content = (Path(workspace) / "requirements.txt").read_text()
        # Original had flask==2.2.0, requests==2.28.0 — check for newer
        old_pins = ["flask==2.2.0", "requests==2.28.0", "numpy==1.23.0"]
        still_old = [p for p in old_pins if p in content]
        passed = len(still_old) == 0
        return passed, f"Still on old versions: {still_old or 'none'}"
    except Exception as e:
        return False, f"Validation error: {e}"


def _validate_explanation(output: str, workspace: str) -> tuple[bool, str]:
    """Check that output mentions key module names."""
    key_terms = ["auth", "database", "session", "password"]
    found = [t for t in key_terms if t.lower() in output.lower()]
    passed = len(found) >= 3
    return passed, f"Mentioned {len(found)}/{len(key_terms)} key concepts"


TASK_REGISTRY: list[BenchmarkTask] = [
    BenchmarkTask(
        id="add_docstrings",
        name="Add Missing Docstrings",
        description="Add Google-style docstrings to all public functions in the sample project.",
        prompt=(
            "The codebase in the workspace is missing docstrings on most functions. "
            "Add Google-style docstrings to all public functions in auth.py, database.py, and utils.py. "
            "Do not change any logic. Only add docstrings."
        ),
        validator=_validate_docstrings,
        tags=["code-quality", "documentation"],
    ),
    BenchmarkTask(
        id="sync_to_async",
        name="Sync → Async Refactor",
        description="Convert all sync database functions in database.py to async/await.",
        prompt=(
            "The file database.py contains synchronous database functions using sqlite3. "
            "Refactor all functions to use async/await with aiosqlite. "
            "Preserve all function signatures and return types. "
            "Add 'import aiosqlite' at the top."
        ),
        validator=_validate_async,
        tags=["refactor", "async"],
    ),
    BenchmarkTask(
        id="dead_code_cleanup",
        name="Dead Code Cleanup",
        description="Identify and remove dead code (unused functions, constants) from the codebase.",
        prompt=(
            "Analyze the Python files in the workspace and remove all dead code: "
            "unused functions, unused variables, and unreachable code blocks. "
            "Do not remove anything that is actively used. "
            "Add a comment explaining what was removed and why."
        ),
        validator=_validate_dead_code,
        tags=["cleanup", "dead-code"],
    ),
    BenchmarkTask(
        id="dependency_migration",
        name="Dependency Version Migration",
        description="Update requirements.txt to use current stable versions of all dependencies.",
        prompt=(
            "The requirements.txt file contains outdated dependency versions. "
            "Update each dependency to its current stable version. "
            "Do not add or remove packages, only update version pins."
        ),
        validator=_validate_deps,
        tags=["dependencies", "maintenance"],
    ),
    BenchmarkTask(
        id="explain_codebase",
        name="Codebase Explanation",
        description="Generate a comprehensive explanation of the sample project architecture.",
        prompt=(
            "Read the files in the workspace and write a comprehensive technical explanation of: "
            "1. What this project does "
            "2. How the authentication system works "
            "3. The database access patterns used "
            "4. Any security concerns you notice "
            "Output plain markdown."
        ),
        validator=_validate_explanation,
        tags=["documentation", "analysis"],
    ),
]
