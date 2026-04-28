# Makefile — SumoSpace

.PHONY: install install-dev install-all test test-fast lint fmt typecheck clean info help

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e "."

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all,dev]"

install-cloud:
	pip install -e ".[all-cloud,dev]"

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	# Skip tests that require model downloads
	pytest tests/ -v --tb=short -k "not (test_ingest_directory or test_query or test_rerank or test_retrieve)"

test-cov:
	pytest tests/ --cov=sumospace --cov-report=html --cov-report=term-missing

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:
	ruff check sumospace/ tests/

fmt:
	ruff check --fix sumospace/ tests/
	ruff format sumospace/ tests/

typecheck:
	mypy sumospace/ --ignore-missing-imports

# ── CLI shortcuts ─────────────────────────────────────────────────────────────

info:
	sumo info

# ── Smoke test (no API key needed) ───────────────────────────────────────────

smoke:
	python -c "\
import asyncio; \
from sumospace.kernel import SumoKernel, KernelConfig; \
async def t(): \
    k = SumoKernel(KernelConfig(provider='hf', model='default', dry_run=True, verbose=True, require_consensus=False)); \
    async with k: \
        tr = await k.run('List all Python files'); \
        print('SUCCESS:', tr.success); \
        print('INTENT:', tr.intent); \
asyncio.run(t()) \
"

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf .sumo_db/ .pytest_cache/ htmlcov/ dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo "SumoSpace Makefile"
	@echo ""
	@echo "  make install       — Install core (local, no API keys)"
	@echo "  make install-dev   — Install core + dev tools"
	@echo "  make install-all   — Install everything local"
	@echo "  make install-cloud — Install everything including cloud providers"
	@echo ""
	@echo "  make test          — Run full test suite"
	@echo "  make test-fast     — Run tests that don't require model downloads"
	@echo "  make test-cov      — Run tests with coverage report"
	@echo ""
	@echo "  make lint          — Check code with ruff"
	@echo "  make fmt           — Format code with ruff"
	@echo "  make typecheck     — Type check with mypy"
	@echo ""
	@echo "  make smoke         — Quick smoke test (dry-run, no model download)"
	@echo "  make info          — Show installed capabilities"
	@echo "  make clean         — Remove build artifacts and caches"
