# sumospace/benchmarks/standalone.py
"""
SumoSpace Real Benchmark Runner (Package-Bundled)
==================================================
Runs actual LLM inference against real coding tasks.
Produces publishable results with full transparency.

Called via:
    sumo benchmark run --provider ollama --model phi3:mini
    sumo benchmark run --provider ollama --model llama3:8b --modes full,disabled
    sumo benchmark run --task add_docstrings --modes full

Or standalone:
    python -m sumospace.benchmarks.standalone --provider ollama --model phi3:mini
"""

import asyncio
import ast
import json
import shutil
import sys
import time
import traceback
import tempfile
import platform
import importlib.util
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

# ── Configuration ─────────────────────────────────────────────────────────────
# Fixtures ship inside the package
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_project"

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL    = "phi3:mini"
DEFAULT_MODES    = ["disabled", "plan_only", "critique_only", "full"]

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_name:      str
    committee_mode: str
    success:        bool
    score:          float          # 0.0 to 1.0
    duration_s:     float
    error:          str = ""
    tool_failures:  int = 0
    retries:        int = 0
    steps_executed: int = 0
    notes:          str = ""

@dataclass
class BenchmarkRun:
    provider:       str
    model:          str
    hardware:       str
    sumoversion:    str
    started_at:     str
    finished_at:    str = ""
    results:        list[TaskResult] = field(default_factory=list)

# ── Verifier functions ────────────────────────────────────────────────────────
# Each returns (success: bool, score: float, notes: str)

def verify_docstrings(workspace: Path) -> tuple[bool, float, str]:
    """Verify all functions in utils.py have docstrings."""
    target_file = workspace / "utils.py"
    if not target_file.exists():
        return False, 0.0, "utils.py not found"

    try:
        content = target_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, 0.0, f"Syntax error in utils.py: {e}"

    total_funcs = 0
    funcs_with_docs = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            total_funcs += 1
            if ast.get_docstring(node) is not None:
                funcs_with_docs += 1

    if total_funcs == 0:
        return False, 0.0, "No functions found in utils.py"

    score = funcs_with_docs / total_funcs
    success = (funcs_with_docs == total_funcs)
    notes = f"{funcs_with_docs}/{total_funcs} functions have docstrings"
    return success, score, notes


def verify_dead_code(workspace: Path, original_workspace: Path) -> tuple[bool, float, str]:
    """Verify dead code was removed and file is still valid Python."""
    target_file = workspace / "dead_code.py"
    original_file = original_workspace / "dead_code.py"

    if not target_file.exists():
        return False, 0.0, "dead_code.py not found"

    try:
        content = target_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, 0.0, f"Syntax error: {e}"

    dead_names = ["_legacy_hash_password", "_old_format_date", "_deprecated_validate", "base64", "_LEGACY_VERSION", "_DEPRECATED_FLAG"]
    live_names = ["hash_content", "read_config", "write_config", "json", "hashlib"]

    found_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found_names.add(node.module)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found_names.add(target.id)

    dead_found = [n for n in dead_names if n in found_names]
    live_found = [n for n in live_names if n in found_names]

    is_shorter = len(content) < len(original_file.read_text(encoding="utf-8"))

    if not is_shorter:
        return False, 0.0, "File is not shorter than original"

    if len(live_found) < len(live_names):
        missing = set(live_names) - set(live_found)
        return False, 0.0, f"Live code was incorrectly removed: {missing}"

    score = 1.0 - (len(dead_found) / len(dead_names))
    success = (len(dead_found) == 0 and is_shorter and len(live_found) == len(live_names))
    notes = f"Removed {len(dead_names) - len(dead_found)}/{len(dead_names)} dead items"
    return success, score, notes


def verify_async(workspace: Path) -> tuple[bool, float, str]:
    """Verify I/O functions were converted to async."""
    target_file = workspace / "sync_io.py"
    if not target_file.exists():
        return False, 0.0, "sync_io.py not found"

    try:
        content = target_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, 0.0, f"Syntax error: {e}"

    total_funcs = 0
    async_funcs = 0
    has_asyncio = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "asyncio":
                    has_asyncio = True
        elif isinstance(node, ast.FunctionDef):
            total_funcs += 1
        elif isinstance(node, ast.AsyncFunctionDef):
            total_funcs += 1
            async_funcs += 1

    if total_funcs == 0:
        return False, 0.0, "No functions found"

    score = async_funcs / total_funcs
    success = (score >= 0.8 and has_asyncio)
    notes = f"{async_funcs}/{total_funcs} functions are async. Asyncio imported: {has_asyncio}"
    return success, score, notes


def verify_bugs(workspace: Path) -> tuple[bool, float, str]:
    """Run the verifier test functions from buggy.py."""
    target_file = workspace / "buggy.py"
    if not target_file.exists():
        return False, 0.0, "buggy.py not found"

    spec = importlib.util.spec_from_file_location("buggy_test_mod", str(target_file))
    if spec is None or spec.loader is None:
        return False, 0.0, "Failed to load module spec"

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SyntaxError as e:
        return False, 0.0, f"Syntax error: {e}"
    except Exception as e:
        return False, 0.0, f"Execution error on import: {e}"

    if not hasattr(mod, "ALL_TESTS"):
        return False, 0.0, "ALL_TESTS list missing from buggy.py"

    tests = mod.ALL_TESTS
    if not tests:
        return False, 0.0, "No tests found in ALL_TESTS"

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError:
            pass
        except Exception:
            pass

    score = passed / len(tests)
    success = (passed == len(tests))
    notes = f"{passed}/{len(tests)} tests passed"
    return success, score, notes


def verify_explanation(response_text: str, workspace: Path) -> tuple[bool, float, str]:
    """Verify the explanation is substantive and mentions real function names."""
    if not response_text:
        return False, 0.0, "Empty response"

    word_count = len(response_text.split())
    if word_count < 100:
        return False, 0.0, f"Response too short ({word_count} words)"

    if response_text.strip().startswith("def ") or response_text.strip().startswith("import "):
        return False, 0.0, "Response looks like raw code rather than an explanation"

    expected_symbols = ["calculate_discount", "safe_divide", "fetch_user"]
    found_symbols = [sym for sym in expected_symbols if sym in response_text]

    score = 1.0 if len(found_symbols) > 0 and word_count >= 100 else 0.0
    success = score == 1.0
    notes = f"{word_count} words, {len(found_symbols)} key symbols mentioned"
    return success, score, notes


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = [
    {
        "name":     "add_docstrings",
        "file":     "utils.py",
        "prompt":   "Add a docstring to every function in utils.py that is missing one. Do not change any logic.",
        "verifier": verify_docstrings,
        "needs_original": False
    },
    {
        "name":     "dead_code_removal",
        "file":     "dead_code.py",
        "prompt":   "Remove all dead code from dead_code.py. Dead code means: functions that are never called and imports that are never used.",
        "verifier": verify_dead_code,
        "needs_original": True
    },
    {
        "name":     "sync_to_async",
        "file":     "sync_io.py",
        "prompt":   "Refactor all I/O functions in sync_io.py to use async/await. Replace time.sleep with asyncio.sleep and requests with httpx async client.",
        "verifier": verify_async,
        "needs_original": False
    },
    {
        "name":     "fix_bugs",
        "file":     "buggy.py",
        "prompt":   "Find and fix all bugs in buggy.py. The file contains: unhandled division by zero, an off-by-one error in a loop, a wrong comparison operator, and a missing return statement.",
        "verifier": verify_bugs,
        "needs_original": False
    },
    {
        "name":     "explain_codebase",
        "file":     None,
        "prompt":   "Explain what this codebase does. Describe the main purpose, key functions, data flow, and any issues. Write for a new developer joining the project.",
        "verifier": verify_explanation,
        "needs_original": False
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────

async def check_provider(provider: str, model: str):
    if provider == "ollama":
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get("http://localhost:11434/api/tags", timeout=5)
                r.raise_for_status()
        except Exception:
            print("ERROR: Ollama not running. Start with: ollama serve")
            sys.exit(1)

async def run_single_task(
    task: dict,
    committee_mode: str,
    provider: str,
    model: str,
) -> TaskResult:
    """
    Run one task in one committee mode.
    Creates isolated workspace, runs kernel, verifies output.
    Never modifies fixture files.
    """
    from sumospace.settings import SumoSettings
    from sumospace.kernel import SumoKernel

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ws = Path(tmpdir)
        shutil.copytree(FIXTURES_DIR, tmp_ws, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".sumo_db"))

        settings = SumoSettings(
            provider=provider,
            model=model,
            workspace=str(tmp_ws),
            committee_enabled=(committee_mode != "disabled"),
            committee_mode=committee_mode if committee_mode != "disabled" else "full",
            vector_store="faiss",
            memory_enabled=False,
            rag_enabled=False,
            verbose=False,
            dry_run=False
        )

        start = time.time()
        error_msg = ""
        output_text = ""
        tool_failures = 0
        tool_calls = 0
        steps_executed = 0

        try:
            async with SumoKernel(settings=settings) as kernel:
                trace = await kernel.run(task["prompt"])
                output_text = str(getattr(trace, "final_output", getattr(trace, "final_answer", trace)))

                step_traces = getattr(trace, "step_traces", [])
                steps_executed = len(step_traces)
                called_tools = [s.tool for s in step_traces]
                tool_calls = len(called_tools)
                tool_failures = sum(1 for tc in getattr(trace, "tool_calls", []) if not getattr(tc, "success", True))
        except Exception as e:
            error_msg = f"Kernel error: {e}"

        duration = time.time() - start

        # Verify
        success, score, notes = False, 0.0, "Failed to run verification"
        try:
            if not error_msg:
                if task.get("needs_original"):
                    success, score, notes = task["verifier"](tmp_ws, FIXTURES_DIR)
                elif task["name"] == "explain_codebase":
                    success, score, notes = task["verifier"](output_text, tmp_ws)
                else:
                    success, score, notes = task["verifier"](tmp_ws)
            else:
                notes = error_msg
        except Exception as e:
            error_msg = f"Verifier exception: {e}"
            notes = error_msg

        return TaskResult(
            task_name=task["name"],
            committee_mode=committee_mode,
            success=success,
            score=score,
            duration_s=round(duration, 2),
            error=error_msg,
            tool_failures=tool_failures,
            steps_executed=steps_executed,
            notes=notes
        )

async def run_all(
    provider: str,
    model: str,
    modes: list[str],
    task_filter: str | None = None,
) -> BenchmarkRun:
    """
    Run all tasks across all modes.
    Prints live progress to stdout.
    """
    await check_provider(provider, model)

    from sumospace import __version__

    run = BenchmarkRun(
        provider=provider,
        model=model,
        hardware=detect_hardware(),
        sumoversion=__version__,
        started_at=datetime.now().isoformat(),
    )

    target_tasks = [t for t in TASKS if not task_filter or t["name"] == task_filter]
    total_runs = len(modes) * len(target_tasks)

    print("-" * 60)

    run_idx = 1
    for mode in modes:
        for task in target_tasks:
            sys.stdout.write(f"[{run_idx:02d}/{total_runs:02d}] {task['name']:<20} | {mode:<14} | running...\r")
            sys.stdout.flush()

            result = await run_single_task(task, mode, provider, model)
            run.results.append(result)

            status_char = "✓" if result.success else "✗"
            score_pct = f"{result.score * 100:.1f}%"
            sys.stdout.write(f"[{run_idx:02d}/{total_runs:02d}] {task['name']:<20} | {mode:<14} | {status_char} {score_pct:<6} | {result.duration_s:>5.1f}s | {result.notes}\n")
            sys.stdout.flush()

            run_idx += 1

    run.finished_at = datetime.now().isoformat()
    return run

# ── Reporting ─────────────────────────────────────────────────────────────────

def generate_report(run: BenchmarkRun) -> str:
    """Generate a markdown report from a BenchmarkRun."""

    modes = []
    tasks = []

    for r in run.results:
        if r.committee_mode not in modes:
            modes.append(r.committee_mode)
        if r.task_name not in tasks:
            tasks.append(r.task_name)

    mode_summaries = []
    for mode in modes:
        mode_results = [r for r in run.results if r.committee_mode == mode]
        if not mode_results:
            continue
        success_count = sum(1 for r in mode_results if r.success)
        avg_dur = sum(r.duration_s for r in mode_results) / len(mode_results)
        tool_failures = sum(r.tool_failures for r in mode_results)
        mode_summaries.append({
            "mode": mode,
            "success_rate": f"{(success_count/len(mode_results))*100:.0f}% ({success_count}/{len(mode_results)})",
            "avg_dur": f"{avg_dur:.1f}s",
            "failures": tool_failures
        })

    summary_table = "\n".join([
        f"| `{s['mode']}` | {s['success_rate']} | {s['avg_dur']} | {s['failures']} |"
        for s in mode_summaries
    ])

    grid_header = "| Task | " + " | ".join(modes) + " |"
    grid_divider = "|------|" + "|".join(["---"] * len(modes)) + "|"

    grid_rows = []
    for t in tasks:
        row = [f"**{t}**"]
        for m in modes:
            res = next((r for r in run.results if r.task_name == t and r.committee_mode == m), None)
            if res:
                row.append(f"{res.score*100:.0f}%")
            else:
                row.append("-")
        grid_rows.append("| " + " | ".join(row) + " |")

    grid = "\n".join([grid_header, grid_divider] + grid_rows)

    date_str = datetime.fromisoformat(run.started_at).strftime("%Y-%m-%d")

    report = f"""# SumoSpace Benchmark Results

**Model:** {run.model}  
**Provider:** {run.provider}  
**Hardware:** {run.hardware}  
**SumoSpace:** {run.sumoversion}  
**Date:** {date_str}  

## Summary

| Committee Mode | Success Rate | Avg Duration | Tool Failures |
|----------------|-------------|--------------|---------------|
{summary_table}

## Results by Task

{grid}

## Methodology

Tasks run against bundled fixtures in `sumospace/benchmarks/fixtures/sample_project/`.
Each task run uses an isolated temporary workspace.
Verification is deterministic — no LLM used for scoring.
Scores represent task-specific metrics (see `docs/benchmarks.md`).
"""
    return report

# ── Hardware detection ────────────────────────────────────────────────────────

def detect_hardware() -> str:
    """Detect and describe the current hardware."""
    sys_info = platform.system()
    machine = platform.machine()
    processor = platform.processor() or machine

    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3))
        return f"{sys_info} {processor}, {ram_gb}GB RAM"
    except ImportError:
        return f"{sys_info} {processor}"

# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SumoSpace Real Benchmark Runner")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--modes",    default=",".join(DEFAULT_MODES),
                        help="Comma-separated: disabled,plan_only,critique_only,full")
    parser.add_argument("--task",     default=None,
                        help="Run only one task by name")
    parser.add_argument("--output",   default=None,
                        help="Output file path (default: ./benchmark_results/TIMESTAMP.md)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════╗
║         SumoSpace Real Benchmark Runner              ║
╠══════════════════════════════════════════════════════╣
║  Provider:  {args.provider:<41}║
║  Model:     {args.model:<41}║
║  Modes:     {', '.join(modes):<41}║
║  Hardware:  {detect_hardware():<41}║
╚══════════════════════════════════════════════════════╝

WARNING: This runs real LLM inference.
Estimated time: {len(modes) * len(TASKS) * 2}-{len(modes) * len(TASKS) * 4} minutes on Ollama + phi3:mini

Press Ctrl+C to cancel.
Starting in 3 seconds...
""")

    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    run = asyncio.run(run_all(
        provider=args.provider,
        model=args.model,
        modes=modes,
        task_filter=args.task,
    ))

    report = generate_report(run)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else results_dir / f"benchmark_{timestamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(asdict(run), indent=2))

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"Raw JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
