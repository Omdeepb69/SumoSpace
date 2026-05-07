# Benchmarks

SumoSpace ships with a reproducible evaluation framework. Every score is produced by a deterministic code-level verifier — no LLM grades another LLM's output.

---

## What We Benchmark

The suite contains 5 tasks, each targeting a distinct class of autonomous coding work. All tasks run against the same fixture codebase — a small Python project in `sumospace/benchmarks/fixtures/sample_project/`.

| # | Task | File | What the agent must do |
|---|---|---|---|
| 1 | `add_docstrings` | `utils.py` | Add a docstring to every undocumented function |
| 2 | `dead_code_removal` | `dead_code.py` | Remove unused functions and unused imports |
| 3 | `sync_to_async` | `sync_io.py` | Refactor all I/O functions to async/await |
| 4 | `fix_bugs` | `buggy.py` | Find and fix 4 distinct bugs |
| 5 | `explain_codebase` | *(all)* | Write a developer explanation of the codebase |

**Why these tasks?** They cover the two main agent capabilities:

- **Write tasks** (1–4) — the agent must produce a file mutation that passes a code-level check.
- **Read tasks** (5) — the agent must produce a substantive, accurate explanation of real code.

---

## How We Verify

**No LLM scores LLM output.** Every verifier is a pure Python function that inspects the AST or file contents directly.

### Example: `add_docstrings` verifier

This is the exact code that produces the score:

```python title="sumospace/benchmarks/standalone.py"
import ast

def verify_docstrings(workspace: Path) -> tuple[bool, float, str]:
    """Verify all functions in utils.py have docstrings."""
    target_file = workspace / "utils.py"
    if not target_file.exists():
        return False, 0.0, "utils.py not found"

    content = target_file.read_text(encoding="utf-8")
    tree = ast.parse(content)

    total_funcs = 0
    funcs_with_docs = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            total_funcs += 1
            if ast.get_docstring(node) is not None:
                funcs_with_docs += 1

    score = funcs_with_docs / total_funcs
    success = (funcs_with_docs == total_funcs)
    notes = f"{funcs_with_docs}/{total_funcs} functions have docstrings"
    return success, score, notes
```

The `score` is a float from `0.0` to `1.0`. A score of `0.75` means 6 out of 8 functions received docstrings. `success` is only `True` when all functions have docstrings.

This approach means:

- Results are **reproducible** — you can run the same benchmark on different hardware and get the same verdict.
- Results are **auditable** — you can read the verifier code and understand exactly what passed.
- Results are **honest** — a partial completion gets a partial score, not a binary pass/fail.

---

## Run It Yourself

```bash
# Install
pip install sumospace

# Pull a model
ollama pull llama3:8b

# Run full benchmark (~2-3 hours on CPU, ~30 min on GPU)
sumo benchmark run --provider ollama --model llama3:8b

# Run one task only (~15 minutes)
sumo benchmark run --task add_docstrings --modes disabled,full --provider ollama --model llama3:8b

# View a saved result
cat benchmark_results/benchmark_*.md

# Re-render a JSON result as Markdown
sumo benchmark report benchmark_results/benchmark_20260507_120000.json
```

All results are saved to `./benchmark_results/` as both `.md` (human-readable) and `.json` (machine-readable).

---

## Hardware Requirements

| Hardware | Estimated time | Recommended model |
|---|---|---|
| CPU only, 8 GB RAM | 2–3 hours | `phi3:mini` |
| CPU only, 16 GB RAM | ~90 minutes | `llama3:8b` |
| GPU (RTX 3090+) | 20–30 minutes | `llama3:8b` |
| GPU (A100) | 8–12 minutes | `llama3:70b` |

!!! warning "Small models"
    Models under 7B parameters (e.g. `phi3:mini`) lack the instruction-following fidelity to reliably pass committee JSON schema validation. Use `--modes disabled` to evaluate small models on raw task completion, bypassing the committee pipeline.

---

## Results

!!! info "Community validation in progress"
    We are collecting benchmark results across hardware configurations before publishing official numbers. We do not publish scores we cannot independently reproduce.

    Once results are available, they will appear here in this format:

    | Task | disabled | plan_only | critique_only | full |
    |---|---|---|---|---|
    | `add_docstrings` | — | — | — | — |
    | `dead_code_removal` | — | — | — | — |
    | `sync_to_async` | — | — | — | — |
    | `fix_bugs` | — | — | — | — |
    | `explain_codebase` | — | — | — | — |

---

## Community Results

Run the benchmark and share your results:

1. Run the benchmark:
    ```bash
    sumo benchmark run --provider ollama --model <your-model>
    ```

2. Open a GitHub issue titled `[Benchmark] <model> on <hardware>`.

3. Attach the full output `.md` file and the `.json` file from `./benchmark_results/`.

Please include:

- Model name and size
- Hardware (CPU / GPU, RAM)
- SumoSpace version (`sumo --version`)
- The full output file