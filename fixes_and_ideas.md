# SumoSpace — Road to 100/100
### Complete Fix & Enhancement Roadmap for Industry-Grade Adoption

> **Current score: 52/100** → **Target: 100/100**
> This document is the single source of truth for every bug fix, security patch, architectural improvement, and new feature needed to make SumoSpace a production-grade, industry-adoptable library.

---

## Table of Contents

1. [Critical Bug Fixes](#1-critical-bug-fixes)
2. [High Severity Fixes](#2-high-severity-fixes)
3. [Medium Severity Fixes](#3-medium-severity-fixes)
4. [Low Severity Fixes](#4-low-severity-fixes)
5. [Customizability Overhaul](#5-customizability-overhaul)
6. [New Features & Ideas](#6-new-features--ideas)
7. [Observability & Debugging](#7-observability--debugging)
8. [Testing & Quality](#8-testing--quality)
9. [Documentation & DX](#9-documentation--dx)
10. [Production & Deployment](#10-production--deployment)
11. [Implementation Priority Order](#11-implementation-priority-order)
12. [Score Progression](#12-score-progression)

---

## 1. Critical Bug Fixes

### 1.1 — Shell Tool Sandbox (Security)
**File:** `sumospace/tools.py` — Lines 163–169
**Problem:** The blocked-command list is a static set of 4 strings. `rm -rf /home`, `rm -rf /*`, `chmod -R 777 /`, fork bombs with slight variation — all bypass it trivially.

**Fix:**
```python
import re
import shlex

BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-[rf]{1,2}\s+[/~]"),         # rm -rf on root/home
    re.compile(r"rm\s+-[rf]{1,2}\s+\*"),            # rm -rf *
    re.compile(r"mkfs"),                             # format disk
    re.compile(r"dd\s+if=/dev/zero"),               # zero out disk
    re.compile(r":\(\)\{.*\}"),                      # fork bomb
    re.compile(r">\s*/dev/sd[a-z]"),                # overwrite block device
    re.compile(r"chmod\s+-R\s+[0-7]*7[0-7]*\s+/"), # recursive chmod on /
    re.compile(r"mv\s+.*\s+/dev/null"),              # destroy files via mv
    re.compile(r"wget.*\|\s*sh"),                    # remote code execution
    re.compile(r"curl.*\|\s*sh"),                    # remote code execution
    re.compile(r"sudo\s+rm"),                        # sudo delete
    re.compile(r"shred\s+"),                         # shred files
]

def _is_blocked(command: str) -> tuple[bool, str]:
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return True, f"Blocked pattern: {pattern.pattern}"
    return False, ""
```

**Additionally** — add a `sandbox` mode using Docker or `firejail`:
```python
@dataclass
class ShellConfig:
    timeout: int = 60
    sandbox: bool = False                    # wrap in docker --rm
    sandbox_image: str = "python:3.11-slim"
    allowed_commands: list[str] | None = None  # if set, allowlist only
    max_output_bytes: int = 1_000_000        # 1MB cap on output
    working_dir_only: bool = True            # restrict cwd to workspace
```

---

### 1.2 — Silent Random Vector Fallback (Data Corruption)
**File:** `sumospace/ingest.py` — `_fallback_embed` method

**Problem:** When both sentence-transformers and ChromaDB's default embedder fail, the code silently returns random unit vectors. Every RAG query returns garbage results with no error.

**Fix — fail loudly:**
```python
async def _fallback_embed(self, texts: list[str]) -> list[list[float]]:
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        fn = DefaultEmbeddingFunction()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, fn, texts)
        return [list(v) for v in result]
    except Exception as e:
        raise IngestError(
            f"All embedding backends failed: {e}\n"
            "Install sentence-transformers: pip install sumospace\n"
            "Or set embedding_provider='google'/'openai' with the matching API key."
        ) from e
```

---

## 2. High Severity Fixes

### 2.1 — Critic Return Type Mismatch
**File:** `sumospace/committee.py` — `CriticAgent.critique` return signature

**Problem:** Method declares `-> tuple[str, str, list[str], list[str]]` but returns 5 elements. All callers unpack 5 values, so it works by accident, but the annotation is a lie.

**Fix:**
```python
async def critique(
    self,
    plan: ExecutionPlan,
    task: str,
) -> tuple[str, str, list[str], list[str], str]:
    """
    Returns:
        verdict (str): "approve" | "revise" | "reject"
        reason (str): one-sentence explanation
        risks (list[str]): identified risks
        blockers (list[str]): must-fix issues
        raw (str): raw LLM output for audit logging
    """
```

---

### 2.2 — Resolver Silently Approves Rejected Plans
**File:** `sumospace/committee.py` — `ResolverAgent.resolve` — Lines ~185–195

**Problem:** When the resolver's JSON is unparseable (common with small local models), the code approves the original plan with a warning comment. This bypasses the critic entirely.

**Fix:**
```python
except Exception as e:
    # A parse failure on a flagged plan is NOT approval.
    # Better to halt than to execute a critic-rejected plan.
    return original_plan, False, "", (
        f"Resolver output unparseable ({e}). "
        "Refusing to approve a critic-flagged plan with unverifiable resolution. "
        "Retry or use --no-consensus to bypass."
    )
```

**Additionally** — add a retry with stricter JSON-only prompt before giving up:
```python
# In ResolverAgent.__init__
self.max_parse_retries: int = 2

# In resolve(), wrap the LLM call in a retry loop
# that adds "Output ONLY valid JSON, no other text" to the prompt
```

---

### 2.3 — trace.success Never Set True in Normal Path
**File:** `sumospace/kernel.py` — `run()` method

**Problem:** `trace.success` is only explicitly set to `True` in the `dry_run` branch. In normal execution, `_execute_plan` sets it based on step results, but any exception after that (in `_synthesise` or memory persist) leaves it in whatever state it was.

**Fix — explicit success gate:**
```python
# After _execute_plan and _synthesise both succeed:
try:
    if not trace.final_answer:
        trace.final_answer = await self._synthesise(task, trace, full_context)

    await self._memory.add("user", task)
    await self._memory.add("assistant", trace.final_answer)

    # Only mark success after EVERYTHING succeeds
    trace.success = True

except Exception as e:
    trace.success = False
    trace.error = f"Post-execution failure: {e}"
    trace.final_answer = trace.final_answer or f"Execution completed but synthesis failed: {e}"
```

---

### 2.4 — Gemini Thread-Safety
**File:** `sumospace/providers.py` — `GeminiProvider`

**Problem:** `self._client` (`GenerativeModel`) is shared across concurrent `run_in_executor` calls. Not thread-safe.

**Fix — use async API and per-call model instantiation:**
```python
async def complete(
    self, user: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048
) -> str:
    import google.generativeai as genai
    # Fresh model per call — avoids shared state under concurrency
    model = genai.GenerativeModel(
        model_name=self.model,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    prompt = f"{system}\n\n{user}" if system else user
    # Use async version
    response = await model.generate_content_async(prompt)
    return response.text
```

---

### 2.5 — scope.py Registry Race Condition
**File:** `sumospace/scope.py` — `_register_session`

**Problem:** Read-modify-write on `registry.json` is not atomic. Two concurrent processes writing for the same user corrupt the file.

**Fix — file locking:**
```python
import fcntl
from contextlib import contextmanager

@contextmanager
def _locked_file(path: Path, mode: str = "r+"):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create if missing
    if not path.exists():
        path.write_text("{}", encoding="utf-8")
    with open(path, mode, encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def _register_session(self, user_id: str, session_id: str):
    rpath = self._registry_path(user_id)
    with _locked_file(rpath) as f:
        try:
            registry = json.load(f)
        except json.JSONDecodeError:
            registry = {}
        if session_id not in registry:
            registry[session_id] = datetime.now(timezone.utc).isoformat()
            f.seek(0)
            json.dump(registry, f, indent=2, sort_keys=True)
            f.truncate()
```

> **Note for Windows users:** `fcntl` is POSIX-only. Add a `filelock` package as fallback:
> `pip install filelock` and use `FileLock(rpath.with_suffix(".lock"))`.

---

## 3. Medium Severity Fixes

### 3.1 — PatchFileTool Temp File Leaks
**File:** `sumospace/tools.py` — `PatchFileTool.run`

```python
async def run(self, path: str, patch: str, **_) -> ToolResult:
    patch_path = orig_path = None  # Initialize before try
    start = time.monotonic()
    try:
        original = Path(path).read_text(encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        ) as pf:
            pf.write(patch)
            patch_path = pf.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".orig", delete=False, encoding="utf-8"
        ) as orig_f:
            orig_f.write(original)
            orig_path = orig_f.name
        # ... rest of logic
    except Exception as e:
        return ToolResult(tool=self.name, success=False, output="", error=str(e))
    finally:
        for p in [patch_path, orig_path]:
            if p:  # Guard against NameError
                try:
                    os.unlink(p)
                except OSError:
                    pass
```

---

### 3.2 — Consolidate chroma_path / chroma_base
**File:** `sumospace/kernel.py` — `KernelConfig`

**Problem:** Two fields, `chroma_path` and `chroma_base`, both default to `".sumo_db"` and confusingly overlap.

**Fix — single field, always routed through ScopeManager:**
```python
@dataclass
class KernelConfig:
    # Single source of truth for DB location
    chroma_base: str = ".sumo_db"

    # Remove chroma_path entirely.
    # ScopeManager always resolves the final path.
    # When user_id="" (no scope), resolves to chroma_base/default/persistent.db
```

Update `ScopeManager.resolve()` to handle the no-scope case:
```python
def resolve(self, user_id: str = "", ...) -> str:
    if not user_id:
        # No isolation — use shared default path
        path = self.chroma_base / "default" / "persistent.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    # ... existing logic
```

---

### 3.3 — MemoryManager session_id Mismatch
**File:** `sumospace/memory.py` — `MemoryManager.__init__`

```python
def __init__(self, ..., session_id: str = "", scope_manager=None, ...):
    # Generate FIRST, then pass to scope manager
    self.session_id = session_id or _generate_session_id()

    if scope_manager is not None:
        chroma_path = scope_manager.resolve(
            user_id=user_id,
            session_id=self.session_id,   # ← use self.session_id, not raw session_id
            project_id=project_id,
        )
```

---

### 3.4 — OllamaProvider Missing Error Handling
**File:** `sumospace/providers.py` — `OllamaProvider.complete`

```python
async def complete(self, user: str, system: str = "", ...) -> str:
    messages = []
    if system:
        messages.insert(0, {"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    try:
        resp = await self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except httpx.HTTPStatusError as e:
        raise ProviderError(
            f"Ollama returned HTTP {e.response.status_code}. "
            f"Is the model '{self.model}' loaded? Run: ollama pull {self.model}"
        ) from e
    except httpx.ConnectError:
        raise ProviderNotConfiguredError(
            f"Cannot reach Ollama at {self.base_url}. Run: ollama serve"
        )
```

---

### 3.5 — RuleBasedClassifier Overlapping Patterns
**File:** `sumospace/classifier.py`

Add priority tiers and exclusion guards:
```python
# Tier 1 — Highly specific, checked first
TIER1_RULES = [
    (re.compile(r"\b(write|add|create|generate)\b.{0,20}\b(test|tests|unit test|pytest|spec)\b", re.I),
     Intent.WRITE_TESTS, 0.92),
    (re.compile(r"\b(docker|container|dockerfile|compose|image|pod|k8s|kubernetes)\b", re.I),
     Intent.DOCKER_OPERATION, 0.92),
]

# Tier 2 — General, only checked if Tier 1 has no match
TIER2_RULES = [
    (re.compile(r"\b(write|create|implement|build)\b.{0,30}\b(function|class|module|script|api)\b", re.I),
     Intent.WRITE_CODE, 0.85),
    # WRITE_FILE only fires if WRITE_TESTS and WRITE_CODE didn't
    (re.compile(r"\b(write|save|create|update|edit|modify)\b.{0,20}\b(file|\.py|\.js|\.md)\b", re.I),
     Intent.WRITE_FILE, 0.78),
]

def classify(self, text: str) -> ClassificationResult | None:
    for tier in [self.TIER1_RULES, self.TIER2_RULES]:
        best = max(
            ((p, i, c) for p, i, c in tier if p.search(text)),
            key=lambda x: x[2],
            default=None
        )
        if best:
            pattern, intent, confidence = best
            return ClassificationResult(
                intent=intent, confidence=confidence,
                needs_execution=intent in EXECUTION_INTENTS,
                needs_web=intent in WEB_INTENTS,
                needs_retrieval=intent in RETRIEVAL_INTENTS,
                reasoning=f"rule-tier-match: {confidence:.2f}",
            )
    return None
```

---

## 4. Low Severity Fixes

### 4.1 — kernel.py shutdown() Cleanup
```python
async def shutdown(self):
    """Graceful shutdown — release all held resources."""
    try:
        if self._memory and self._memory.episodic._client:
            # ChromaDB doesn't have an explicit close() — 
            # setting to None releases the file lock
            self._memory.episodic._client = None
            self._memory.episodic._collection = None

        if hasattr(self._provider, '_client') and self._provider._client:
            await self._provider._client.aclose()

    except Exception:
        pass  # Best-effort cleanup
    finally:
        self._initialized = False
        self._provider = self._classifier = self._committee = None
        self._tools = self._memory = self._ingestor = self._rag = None

        if self.config.verbose:
            console.print("[dim]Kernel shutdown complete[/dim]")
```

---

### 4.2 — DockerTool Shell Injection via env Values
```python
async def run_container(self, image: str, ..., env: dict | None = None, ...) -> ToolResult:
    parts = ["docker run"]
    if rm:
        parts.append("--rm")
    if ports:
        for host, container in ports.items():
            parts.append(f"-p {shlex.quote(str(host))}:{shlex.quote(str(container))}")
    if env:
        for k, v in env.items():
            # shlex.quote prevents injection via values like: $(rm -rf /)
            parts.append(f"-e {shlex.quote(str(k))}={shlex.quote(str(v))}")
    parts.append(shlex.quote(image))
    if command:
        parts.append(command)
    return await self._shell.run(command=" ".join(parts))
```

---

### 4.3 — Delete or Integrate scraper.py
Either:
```python
# Option A: Integrate into FetchURLTool as a proper scraping backend
# sumospace/tools.py — FetchURLTool.run()
# Replace the basic httpx fetch with scraper.py logic for JS-rendered pages

# Option B: Just delete it
# git rm sumospace/scraper.py
```

---

## 5. Customizability Overhaul

This is the largest architectural addition. The goal: **every behavior in SumoSpace should be swappable without subclassing the kernel.**

---

### 5.1 — Unified Settings System

Create `sumospace/settings.py` — a single Pydantic Settings class that reads from env, `.env` file, and explicit kwargs, with full validation and documentation:

```python
# sumospace/settings.py
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Literal

class SumoSettings(BaseSettings):
    """
    Master settings for SumoSpace.
    All values can be set via:
      1. Environment variables (SUMO_<FIELD_NAME> in uppercase)
      2. .env file
      3. Explicit kwargs: SumoSettings(provider="ollama")
      4. KernelConfig (takes precedence over env)
    """

    model_config = {"env_prefix": "SUMO_", "env_file": ".env", "extra": "ignore"}

    # ── Provider ──────────────────────────────────────────────────────────────
    provider: Literal["hf", "ollama", "gemini", "openai", "anthropic", "vllm", "auto"] = "hf"
    model: str = "default"
    provider_timeout: int = Field(300, description="LLM call timeout in seconds")
    provider_max_retries: int = Field(3, description="Retry count for failed LLM calls")
    provider_retry_delay: float = Field(1.0, description="Seconds between retries")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: Literal["local", "google", "openai"] = "local"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = Field(32, description="Batch size for embedding calls")
    embedding_cache: bool = Field(True, description="Cache embeddings to avoid re-computation")

    # ── Inference (HuggingFace) ───────────────────────────────────────────────
    hf_load_in_4bit: bool = False
    hf_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    hf_trust_remote_code: bool = False

    # ── Inference (Ollama) ────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_auto_pull: bool = True

    # ── Inference (vLLM) ─────────────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str = "EMPTY"

    # ── Execution ─────────────────────────────────────────────────────────────
    require_consensus: bool = True
    max_plan_steps: int = Field(12, description="Max steps the Planner can propose")
    execution_timeout: int = Field(120, description="Total execution timeout in seconds")
    step_timeout: int = Field(60, description="Per-step tool timeout in seconds")
    dry_run: bool = False
    verbose: bool = True
    max_retries: int = 3

    # ── Committee ─────────────────────────────────────────────────────────────
    committee_temperature: float = Field(0.1, description="LLM temperature for committee agents")
    committee_max_tokens: int = Field(2048, description="Max tokens per committee response")
    planner_max_tokens: int = Field(2048)
    critic_max_tokens: int = Field(1024)
    resolver_max_tokens: int = Field(2048)
    committee_parse_retries: int = Field(2, description="Retries on JSON parse failure")

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_top_k_candidates: int = Field(20, description="Vector search candidates before reranking")
    rag_top_k_final: int = Field(5, description="Final chunks after reranking")
    rag_use_reranker: bool = True
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_max_context_chars: int = Field(6000, description="Max RAG context fed to LLM")
    rag_chunk_size: int = Field(512, description="Text chunk size in tokens")
    rag_chunk_overlap: int = Field(64, description="Overlap between adjacent chunks")

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_working_size: int = Field(20, description="In-process ring buffer size")
    memory_recall_top_k: int = Field(5, description="Default semantic recall count")
    episodic_memory_enabled: bool = True

    # ── Classifier ────────────────────────────────────────────────────────────
    classifier_rule_threshold: float = Field(0.72, description="Min confidence for rule-based hit")
    classifier_nli_threshold: float = Field(0.65, description="Min confidence for NLI hit")
    classifier_nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # ── Storage & Scope ───────────────────────────────────────────────────────
    chroma_base: str = ".sumo_db"
    scope_level: Literal["user", "session", "project", "none"] = "none"
    user_id: str = ""
    session_id: str = ""
    project_id: str = ""
    max_chunks_per_scope: int | None = None

    # ── Shell Tool ────────────────────────────────────────────────────────────
    shell_timeout: int = Field(60, description="Default shell command timeout")
    shell_sandbox: bool = Field(False, description="Run shell commands in Docker sandbox")
    shell_sandbox_image: str = "python:3.11-slim"
    shell_max_output_bytes: int = Field(1_000_000, description="Max shell output size")
    shell_allowed_commands: list[str] | None = Field(None, description="Allowlist (None = all allowed)")

    # ── Web Search ────────────────────────────────────────────────────────────
    web_search_max_results: int = 5
    web_fetch_timeout: int = 20
    web_fetch_max_chars: int = 10000

    # ── Audit & Observability ─────────────────────────────────────────────────
    audit_log_enabled: bool = True
    audit_log_path: str = ".sumo_db/audit"
    telemetry_enabled: bool = False
    telemetry_endpoint: str = ""           # OpenTelemetry OTLP endpoint
    trace_committee: bool = True           # Log full planner/critic/resolver output

    # ── Plan Cache ────────────────────────────────────────────────────────────
    plan_cache_enabled: bool = False
    plan_cache_ttl_hours: float = 24.0
    plan_cache_path: str = ".sumo_db/plan_cache"

    # ── API Keys (read from env) ──────────────────────────────────────────────
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        valid = {"hf", "ollama", "gemini", "openai", "anthropic", "vllm", "auto"}
        if v not in valid:
            raise ValueError(f"provider must be one of {valid}, got '{v}'")
        return v

    @field_validator("committee_temperature", "classifier_rule_threshold")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0 and 2, got {v}")
        return v

    @classmethod
    def from_env(cls) -> "SumoSettings":
        """Load settings from environment + .env file."""
        return cls()

    @classmethod
    def from_file(cls, path: str) -> "SumoSettings":
        """Load settings from a specific .env or .toml file."""
        return cls(_env_file=path)

    def to_kernel_config(self) -> "KernelConfig":
        """Convert to KernelConfig for backward compatibility."""
        from sumospace.kernel import KernelConfig
        return KernelConfig(**{
            k: v for k, v in self.model_dump().items()
            if k in KernelConfig.__dataclass_fields__
        })
```

**Usage:**
```python
# Via environment variables
export SUMO_PROVIDER=ollama
export SUMO_RAG_TOP_K_FINAL=10
export SUMO_SHELL_SANDBOX=true

# Via code
settings = SumoSettings(provider="vllm", rag_use_reranker=False, shell_sandbox=True)
kernel = SumoKernel(settings.to_kernel_config())

# Via config file
settings = SumoSettings.from_file("my_project.env")
```

---

### 5.2 — Plugin System for Tools

Every tool should be registerable from outside the library:

```python
# sumospace/tools.py — Add to BaseTool
from typing import ClassVar

class BaseTool:
    name: str = "base"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""

    # JSON Schema for parameters — enables validation + LLM-readable docs
    schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Tool tags for filtering and routing
    tags: ClassVar[list[str]] = []

    async def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def validate_params(self, kwargs: dict) -> tuple[bool, str]:
        """Validate kwargs against schema. Returns (valid, error_message)."""
        try:
            import jsonschema
            jsonschema.validate(kwargs, self.schema)
            return True, ""
        except ImportError:
            return True, ""  # Skip if jsonschema not installed
        except jsonschema.ValidationError as e:
            return False, str(e.message)

    def describe(self) -> dict:
        """Full tool description for LLM prompt construction."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "schema": self.schema,
            "tags": self.tags,
        }
```

**Plugin registration via entry points** (`pyproject.toml`):
```toml
[project.entry-points."sumospace.tools"]
my_tool = "my_package.tools:MyCustomTool"
```

**Auto-discovery in ToolRegistry:**
```python
class ToolRegistry:
    def _discover_plugins(self):
        """Auto-load tools registered via entry points."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="sumospace.tools")
            for ep in eps:
                try:
                    tool_cls = ep.load()
                    self.register(tool_cls())
                except Exception as e:
                    console.print(f"[yellow]Plugin load failed: {ep.name}: {e}[/yellow]")
        except Exception:
            pass
```

---

### 5.3 — Custom Prompt Templates

Allow users to override every system prompt used by committee agents:

```python
# sumospace/prompts.py
from dataclasses import dataclass

@dataclass
class PromptTemplates:
    """
    All system prompts used by SumoSpace agents.
    Override any field to customize behavior.
    """
    planner_system: str = PLANNER_SYSTEM          # Default from committee.py
    critic_system: str = CRITIC_SYSTEM
    resolver_system: str = RESOLVER_SYSTEM
    classifier_system: str = LLMClassifier.SYSTEM
    synthesiser_system: str = (
        "You are a helpful assistant summarising completed tasks.\n"
        "Given the task and tool outputs, write a clear, concise summary of what was done.\n"
        "Be specific. Mention files changed, commands run, or answers found."
    )
    chat_system: str = "You are Sumo, a helpful AI assistant."

    @classmethod
    def from_file(cls, path: str) -> "PromptTemplates":
        """Load custom prompts from a JSON/TOML file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def for_domain(cls, domain: str) -> "PromptTemplates":
        """
        Pre-built prompt sets for common domains.
        domain: "coding" | "devops" | "research" | "data_science"
        """
        presets = {
            "coding": cls(
                planner_system=CODING_PLANNER_SYSTEM,
                critic_system=CODING_CRITIC_SYSTEM,
            ),
            "devops": cls(
                planner_system=DEVOPS_PLANNER_SYSTEM,
            ),
        }
        return presets.get(domain, cls())
```

**Usage:**
```python
templates = PromptTemplates.for_domain("devops")
# or
templates = PromptTemplates(
    planner_system="You are an expert DevOps engineer. Plan only using kubectl, helm, and terraform.",
    critic_system="You are a security-focused reviewer. Reject any plan that touches production secrets."
)
kernel = SumoKernel(config, prompt_templates=templates)
```

---

### 5.4 — Lifecycle Hooks / Middleware

```python
# sumospace/hooks.py
from typing import Callable, Awaitable
from dataclasses import dataclass, field

@dataclass
class KernelHooks:
    """
    Async hooks called at each stage of the kernel pipeline.
    Return None to continue, raise to halt.
    """
    # Called before classification
    on_task_received: list[Callable] = field(default_factory=list)

    # Called after classification, before committee
    on_classified: list[Callable] = field(default_factory=list)

    # Called after committee approves a plan, before execution
    on_plan_approved: list[Callable] = field(default_factory=list)

    # Called when committee rejects a plan
    on_plan_rejected: list[Callable] = field(default_factory=list)

    # Called after each step executes
    on_step_complete: list[Callable] = field(default_factory=list)

    # Called when a step fails
    on_step_failed: list[Callable] = field(default_factory=list)

    # Called after full execution completes
    on_task_complete: list[Callable] = field(default_factory=list)

    # Called on any error
    on_error: list[Callable] = field(default_factory=list)

    def add(self, event: str, fn: Callable):
        getattr(self, event).append(fn)

    async def fire(self, event: str, *args, **kwargs):
        for fn in getattr(self, event, []):
            await fn(*args, **kwargs) if asyncio.iscoroutinefunction(fn) else fn(*args, **kwargs)
```

**Usage:**
```python
hooks = KernelHooks()

# Require human approval before any execution
async def human_approval_gate(plan: ExecutionPlan):
    print(f"\nPlan has {len(plan.steps)} steps. Approve? [y/N]")
    if input().strip().lower() != "y":
        raise ConsensusFailedError("User rejected plan")

hooks.add("on_plan_approved", human_approval_gate)

# Log every step to Slack
async def slack_notify(step: StepTrace):
    await slack_client.post(f"Step {step.step_number}: {step.tool} — {step.result.success}")

hooks.add("on_step_complete", slack_notify)

kernel = SumoKernel(config, hooks=hooks)
```

---

### 5.5 — Custom Agent Roles in Committee

Allow users to replace or extend the three-agent committee with custom agents:

```python
# sumospace/committee.py — Add base class
class BaseAgent:
    """Override this to add custom committee agents."""
    role: str = "base"

    def __init__(self, provider, settings: SumoSettings):
        self._provider = provider
        self._settings = settings

    async def run(self, task: str, context: str, **kwargs) -> dict:
        raise NotImplementedError

# Allow injecting custom agents into Committee
class Committee:
    def __init__(
        self,
        provider,
        require_consensus: bool = True,
        custom_agents: list[BaseAgent] | None = None,
        # Replace individual agents
        planner: PlannerAgent | None = None,
        critic: CriticAgent | None = None,
        resolver: ResolverAgent | None = None,
    ):
        self._planner = planner or PlannerAgent(provider)
        self._critic = critic or CriticAgent(provider)
        self._resolver = resolver or ResolverAgent(provider)
        self._custom_agents = custom_agents or []
        self.require_consensus = require_consensus
```

**Example — adding a Security Agent:**
```python
class SecurityAuditAgent(BaseAgent):
    role = "security_auditor"

    async def run(self, task: str, context: str, plan: ExecutionPlan, **kwargs) -> dict:
        # Check plan for security issues before Critic runs
        dangerous_tools = [s for s in plan.steps if s.tool == "shell"]
        if len(dangerous_tools) > 3:
            return {"verdict": "flag", "reason": "Too many shell steps for this task"}
        return {"verdict": "pass"}

committee = Committee(
    provider=provider,
    custom_agents=[SecurityAuditAgent(provider, settings)]
)
```

---

## 6. New Features & Ideas

### 6.1 — vLLM Provider

```python
# sumospace/providers.py
class VLLMProvider(BaseProvider):
    """
    vLLM inference server — OpenAI-compatible API.
    Best for: multi-user deployments, GPU servers, high throughput.
    Requires: pip install sumospace[vllm] + running vLLM server.

    Launch: vllm serve microsoft/Phi-3-mini-4k-instruct --dtype auto
    """
    name = "vllm"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        max_concurrent: int = 10,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None

    async def initialize(self):
        import httpx
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300,
        )
        # Health check
        try:
            r = await self._client.get("/health")
            r.raise_for_status()
        except Exception:
            raise ProviderNotConfiguredError(
                f"vLLM server not reachable at {self.base_url}.\n"
                f"Start with: vllm serve {self.model} --dtype auto"
            )

    async def complete(
        self, user: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        import json
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            },
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if delta := data["choices"][0]["delta"].get("content"):
                        yield delta
```

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
vllm = ["vllm>=0.4.0"]
```

---

### 6.2 — Real Streaming Execution (`stream_run`)

```python
# sumospace/kernel.py
async def stream_run(
    self, task: str, session_id: str | None = None
) -> AsyncIterator[StepTrace | ExecutionTrace]:
    """
    Stream execution step-by-step.
    Yields StepTrace after each tool executes, then final ExecutionTrace.

    Usage:
        async for event in kernel.stream_run("refactor auth.py"):
            if isinstance(event, StepTrace):
                print(f"Step {event.step_number}: {event.tool} — {event.result.success}")
            elif isinstance(event, ExecutionTrace):
                print(f"Done: {event.final_answer}")
    """
    if not self._initialized:
        await self.boot()

    session_id = session_id or uuid.uuid4().hex[:12]
    start = time.monotonic()
    trace = ExecutionTrace(
        task=task, session_id=session_id,
        intent=Intent.GENERAL_QA, classification=None, plan=None,
    )

    # Classify + RAG + Committee (same as run())
    classification = await self._classifier.classify(task)
    trace.intent = classification.intent
    trace.classification = classification

    rag_context = ""
    if classification.needs_retrieval:
        try:
            rag_result = await self._rag.retrieve(task)
            rag_context = rag_result.context
        except Exception:
            pass

    verdict = await self._committee.deliberate(task, context=rag_context)
    trace.plan = verdict.plan

    if not verdict.approved:
        trace.error = verdict.rejection_reason
        trace.success = False
        yield trace
        return

    # Stream each step
    for step in verdict.plan.steps:
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
        yield step_trace   # ← Emit after each step

        if not result.success and step.critical:
            trace.error = f"Critical step failed: {step.tool}"
            trace.success = False
            yield trace
            return

    trace.final_answer = await self._synthesise(task, trace, rag_context)
    await self._memory.add("user", task)
    await self._memory.add("assistant", trace.final_answer)
    trace.success = True
    trace.duration_ms = (time.monotonic() - start) * 1000
    yield trace   # ← Final trace
```

---

### 6.3 — Plan Caching

```python
# sumospace/cache.py
import hashlib
import json
import time
from pathlib import Path
from sumospace.committee import ExecutionPlan, ExecutionStep

class PlanCache:
    """
    Content-addressed cache for approved ExecutionPlans.
    Skips the 3-LLM committee on repeat tasks in the same context.
    """

    def __init__(self, cache_dir: str = ".sumo_db/plan_cache", ttl_hours: float = 24.0):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_hours * 3600

    def _key(self, task: str, context: str) -> str:
        raw = f"{task}|||{context[:500]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, task: str, context: str) -> ExecutionPlan | None:
        path = self._dir / f"{self._key(task, context)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if time.time() - data["cached_at"] > self._ttl:
                path.unlink()
                return None
            return self._deserialize(data["plan"])
        except Exception:
            return None

    def set(self, task: str, context: str, plan: ExecutionPlan):
        path = self._dir / f"{self._key(task, context)}.json"
        path.write_text(json.dumps({
            "cached_at": time.time(),
            "task": task,
            "plan": self._serialize(plan),
        }, indent=2))

    def _serialize(self, plan: ExecutionPlan) -> dict:
        return {
            "task": plan.task,
            "reasoning": plan.reasoning,
            "estimated_duration_s": plan.estimated_duration_s,
            "risks": plan.risks,
            "steps": [
                {
                    "step_number": s.step_number,
                    "tool": s.tool,
                    "description": s.description,
                    "parameters": s.parameters,
                    "expected_output": s.expected_output,
                    "critical": s.critical,
                }
                for s in plan.steps
            ],
        }

    def _deserialize(self, data: dict) -> ExecutionPlan:
        return ExecutionPlan(
            task=data["task"],
            steps=[ExecutionStep(**s) for s in data["steps"]],
            reasoning=data.get("reasoning", ""),
            estimated_duration_s=data.get("estimated_duration_s", 0),
            risks=data.get("risks", []),
            approved=True,
            approval_notes="Restored from cache",
        )

    def invalidate(self, task: str, context: str):
        path = self._dir / f"{self._key(task, context)}.json"
        if path.exists():
            path.unlink()

    def clear(self):
        for p in self._dir.glob("*.json"):
            p.unlink()

    def stats(self) -> dict:
        entries = list(self._dir.glob("*.json"))
        return {"count": len(entries), "size_mb": sum(p.stat().st_size for p in entries) / 1e6}
```

---

### 6.4 — Incremental Re-ingestion

```python
# sumospace/ingest.py — in UniversalIngestor
async def ingest_file(self, path: str, force: bool = False) -> IngestionResult:
    """
    Ingest a file, skipping if unchanged since last ingest.
    Set force=True to re-ingest regardless of modification time.
    """
    p = Path(path)
    content_hash = self._file_hash(p)

    if not force:
        # Check if this file hash is already in ChromaDB metadata
        existing = self._collection.get(
            where={"source_hash": content_hash},
            limit=1,
        )
        if existing["ids"]:
            return IngestionResult(
                source=path, chunks_created=0,
                loader_used="skipped (unchanged)", duration_ms=0,
            )

    # Proceed with full ingest, storing hash in metadata
    # ... existing ingest logic ...
    # Add to each chunk's metadata:
    # {"source": path, "source_hash": content_hash, "ingested_at": time.time()}

def _file_hash(self, path: Path) -> str:
    """SHA256 of file content — used to detect changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

async def ingest_directory(self, path: str, force: bool = False) -> list[IngestionResult]:
    """Pass force=True to re-ingest all files regardless of modification."""
    # ... existing logic, pass force to ingest_file ...
```

---

### 6.5 — Committee Audit Log + Replay

```python
# sumospace/audit.py
import json
from pathlib import Path
from datetime import datetime, timezone

class AuditLogger:
    """
    Persists full execution traces and committee verdicts to JSONL.
    One file per session. Enables replay, debugging, and compliance.
    """

    def __init__(self, audit_dir: str = ".sumo_db/audit"):
        self._dir = Path(audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def log(self, trace: "ExecutionTrace", verdict: "CommitteeVerdict"):
        path = self._dir / f"{trace.session_id}.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": trace.task,
            "session_id": trace.session_id,
            "intent": trace.intent.value,
            "confidence": trace.classification.confidence if trace.classification else None,
            "plan_approved": verdict.approved,
            "rejection_reason": verdict.rejection_reason,
            "planner_output": verdict.planner_output,
            "critic_output": verdict.critic_output,
            "resolver_output": verdict.resolver_output,
            "steps_executed": len(trace.step_traces),
            "steps": [
                {
                    "step_number": s.step_number,
                    "tool": s.tool,
                    "success": s.result.success,
                    "duration_ms": s.duration_ms,
                    "error": s.result.error,
                }
                for s in trace.step_traces
            ],
            "final_answer": trace.final_answer,
            "success": trace.success,
            "duration_ms": trace.duration_ms,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_session(self, session_id: str) -> list[dict]:
        path = self._dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def list_sessions(self) -> list[dict]:
        sessions = []
        for p in sorted(self._dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
            lines = p.read_text().splitlines()
            if lines:
                last = json.loads(lines[-1])
                sessions.append({
                    "session_id": p.stem,
                    "task": last.get("task", ""),
                    "success": last.get("success"),
                    "timestamp": last.get("timestamp"),
                })
        return sessions
```

**CLI command:**
```python
# sumospace/cli.py
@app.command()
def replay(
    session_id: str = typer.Argument(..., help="Session ID to replay"),
    audit_dir: str = typer.Option(".sumo_db/audit", "--audit-dir"),
    provider: str = typer.Option("hf", "--provider"),
):
    """Replay a past approved execution plan without re-deliberating."""
    from sumospace.audit import AuditLogger
    from sumospace.kernel import KernelConfig, SumoKernel

    logger = AuditLogger(audit_dir)
    entries = logger.load_session(session_id)
    if not entries:
        console.print(f"[red]No audit log for session: {session_id}[/red]")
        raise typer.Exit(1)

    last = entries[-1]
    console.print(f"[bold]Replaying:[/bold] {last['task']}")
    console.print(f"Original run: {last['timestamp']} — Success: {last['success']}")
    # ... re-execute the logged plan steps ...
```

---

### 6.6 — Intent-Aware Tool Preloading

```python
# sumospace/kernel.py — in run()
async def _prewarm_tools(self, classification: ClassificationResult):
    """Initialize expensive tools in parallel with committee deliberation."""
    tasks = []
    if classification.needs_web:
        # Ensure httpx client is warm
        tasks.append(self._tools.get("web_search").run(query="warmup", max_results=0))

    if classification.intent.value in ("refactor", "code_review", "explain_code"):
        # Pre-load tree-sitter grammars if available
        try:
            import tree_sitter  # noqa: F401
        except ImportError:
            pass

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

# In run(), gather deliberation and prewarming concurrently:
verdict, _ = await asyncio.gather(
    self._committee.deliberate(task, context=full_context),
    self._prewarm_tools(classification),
)
```

---

### 6.7 — `sumo watch` — Filesystem Trigger

```python
# sumospace/cli.py
@app.command()
def watch(
    path: str = typer.Argument(".", help="Directory to watch"),
    task_template: str = typer.Argument(..., help="Task template, use {file} for changed file"),
    provider: str = typer.Option("hf", "--provider"),
    debounce: float = typer.Option(2.0, "--debounce", help="Seconds to wait after last change"),
    extensions: str = typer.Option(".py,.js,.ts", "--ext", help="Comma-separated file extensions"),
):
    """
    Watch a directory and trigger agent runs on file changes.

    Example:
        sumo watch ./src "Run tests and fix failures in {file}"
        sumo watch . "Review the changes in {file} for security issues"
    """
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    import threading

    exts = set(extensions.split(","))

    class ChangeHandler(FileSystemEventHandler):
        def __init__(self):
            self._timer = None
            self._lock = threading.Lock()

        def on_modified(self, event):
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix not in exts:
                return
            with self._lock:
                if self._timer:
                    self._timer.cancel()
                self._timer = threading.Timer(
                    debounce,
                    self._trigger,
                    args=[str(p)],
                )
                self._timer.start()

        def _trigger(self, filepath: str):
            task = task_template.replace("{file}", filepath)
            console.print(f"\n[cyan]Change detected:[/cyan] {filepath}")
            console.print(f"[bold]Running:[/bold] {task}")
            from sumospace.kernel import KernelConfig, SumoKernel
            config = KernelConfig(provider=provider, verbose=True)
            asyncio.run(_run_task(config, task))

    async def _run_task(config, task):
        async with SumoKernel(config) as kernel:
            trace = await kernel.run(task)
            status = "[green]✓[/green]" if trace.success else "[red]✗[/red]"
            console.print(f"{status} {trace.final_answer[:200]}")

    observer = Observer()
    observer.schedule(ChangeHandler(), path, recursive=True)
    observer.start()
    console.print(f"[bold cyan]Watching:[/bold cyan] {path}  (Ctrl+C to stop)")
    try:
        while True:
            import time as t
            t.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

---

### 6.8 — Structured Tool Schemas + Validation

```python
# sumospace/tools.py — Example: ReadFileTool with schema
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file."
    tags = ["filesystem", "read"]

    schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "File encoding",
            },
        },
        "required": ["path"],
    }
```

**In ToolRegistry.execute() — validate before running:**
```python
async def execute(self, name: str, **kwargs) -> ToolResult:
    tool = self.get(name)
    if not tool:
        return ToolResult(tool=name, success=False, output="",
                          error=f"Tool '{name}' not found.")

    valid, error_msg = tool.validate_params(kwargs)
    if not valid:
        return ToolResult(
            tool=name, success=False, output="",
            error=f"Invalid parameters for '{name}': {error_msg}",
            metadata={"validation_error": True},
        )

    return await tool.run(**kwargs)
```

---

## 7. Observability & Debugging

### 7.1 — OpenTelemetry Tracing

```python
# sumospace/telemetry.py
from contextlib import contextmanager

class SumoTelemetry:
    """
    Optional OpenTelemetry integration.
    Enable with: settings.telemetry_enabled = True
    Configure endpoint: settings.telemetry_endpoint = "http://jaeger:4317"
    """

    def __init__(self, enabled: bool = False, endpoint: str = ""):
        self._enabled = enabled
        self._tracer = None
        if enabled:
            self._setup(endpoint)

    def _setup(self, endpoint: str):
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider = TracerProvider()
            if endpoint:
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
                )
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("sumospace")
        except ImportError:
            pass  # Telemetry is opt-in

    @contextmanager
    def span(self, name: str, **attrs):
        if self._tracer:
            with self._tracer.start_as_current_span(name) as span:
                for k, v in attrs.items():
                    span.set_attribute(k, str(v))
                yield span
        else:
            yield None
```

---

### 7.2 — `sumo logs` CLI Command

```python
@app.command()
def logs(
    session_id: str = typer.Option("", "--session", "-s", help="Filter by session ID"),
    last_n: int = typer.Option(10, "--last", "-n", help="Show last N sessions"),
    audit_dir: str = typer.Option(".sumo_db/audit", "--audit-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Browse past execution logs."""
    from sumospace.audit import AuditLogger
    from rich.table import Table

    logger = AuditLogger(audit_dir)

    if session_id:
        entries = logger.load_session(session_id)
        for entry in entries:
            console.print_json(json.dumps(entry, indent=2))
        return

    sessions = logger.list_sessions()[:last_n]
    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="dim")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Timestamp")

    for s in sessions:
        status = "[green]✓[/green]" if s["success"] else "[red]✗[/red]"
        table.add_row(s["session_id"], s["task"][:60], status, s["timestamp"])

    console.print(table)
```

---

## 8. Testing & Quality

### 8.1 — Committee Integration Tests

```python
# tests/test_committee_integration.py
@pytest.mark.asyncio
async def test_critic_blocks_dangerous_plan(mock_provider):
    """Critic must reject plans that delete files without backup."""
    # Mock provider returns a dangerous plan from Planner
    # and a blocking critique from Critic
    mock_provider.responses = [
        DANGEROUS_PLAN_JSON,   # Planner output
        BLOCKING_CRITIQUE_JSON, # Critic output
    ]
    committee = Committee(mock_provider, require_consensus=True)
    verdict = await committee.deliberate("Delete all temp files")
    assert verdict.approved is False
    assert verdict.rejection_reason

@pytest.mark.asyncio
async def test_resolver_parse_failure_does_not_approve(mock_provider):
    """If resolver outputs garbage JSON, plan stays rejected."""
    mock_provider.responses = [
        VALID_PLAN_JSON,
        REVISE_CRITIQUE_JSON,
        "this is not json at all",  # Resolver garbles output
    ]
    committee = Committee(mock_provider, require_consensus=True)
    verdict = await committee.deliberate("Some task")
    assert verdict.approved is False
```

### 8.2 — Scope Isolation Tests

```python
# tests/test_scope_isolation.py
def test_different_users_get_different_paths(tmp_path):
    scope = ScopeManager(chroma_base=str(tmp_path), level="user")
    path_alice = scope.resolve(user_id="alice")
    path_bob = scope.resolve(user_id="bob")
    assert path_alice != path_bob
    assert "alice" in path_alice
    assert "bob" in path_bob

def test_concurrent_session_registration_no_corruption(tmp_path):
    """Race condition test — multiple threads registering sessions."""
    import threading
    scope = ScopeManager(chroma_base=str(tmp_path), level="session")
    errors = []

    def register():
        try:
            scope.resolve(user_id="alice", session_id=uuid.uuid4().hex)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=register) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Registry should have exactly 20 sessions
    sessions = scope.list_sessions("alice")
    assert len(sessions) == 20
```

### 8.3 — Coverage Targets

| Module | Current | Target |
|---|---|---|
| `kernel.py` | ~0% | 80% |
| `committee.py` | ~0% | 85% |
| `scope.py` | ~30% | 90% |
| `tools.py` | ~60% | 90% |
| `providers.py` | ~0% | 75% |
| `classifier.py` | ~40% | 85% |
| `memory.py` | ~20% | 80% |
| `ingest.py` | ~20% | 75% |

Add to `Makefile`:
```makefile
test-cov-check:
	pytest tests/ --cov=sumospace --cov-fail-under=80
```

---

## 9. Documentation & DX

### 9.1 — API Reference (MkDocs)

```
docs/
├── index.md              ← Overview + quickstart
├── getting-started/
│   ├── installation.md
│   ├── zero-config.md    ← The "works with no API key" story
│   └── first-agent.md
├── guides/
│   ├── customization.md  ← All the SumoSettings fields
│   ├── custom-tools.md   ← Writing and registering plugins
│   ├── custom-prompts.md ← Overriding committee prompts
│   ├── hooks.md          ← Lifecycle hooks
│   ├── multi-tenancy.md  ← ScopeManager deep-dive
│   ├── vllm.md           ← Production deployment with vLLM
│   └── security.md       ← Shell sandboxing, best practices
├── api/
│   ├── kernel.md
│   ├── committee.md
│   ├── tools.md
│   ├── providers.md
│   └── settings.md
└── changelog.md
```

### 9.2 — CHANGELOG.md

Every library needs a changelog. Start it now:

```markdown
# Changelog

## [Unreleased]
### Fixed
- Critical: Shell tool blocked-command list now uses regex patterns
- Critical: Embedding fallback raises IngestError instead of returning random vectors
- High: Resolver no longer approves critic-rejected plans on parse failure
- High: trace.success correctly set in all execution paths
...

### Added
- vLLM provider support
- SumoSettings unified configuration
- Plugin system via entry points
- KernelHooks lifecycle middleware
- Plan caching
- Audit log + replay
- sumo watch command
- sumo logs command
- Real streaming via stream_run()
```

### 9.3 — CONTRIBUTING.md

```markdown
# Contributing to SumoSpace

## Adding a new tool
1. Subclass `BaseTool` in `sumospace/tools.py`
2. Define `name`, `description`, `schema`, and `tags`
3. Implement `async def run(self, **kwargs) -> ToolResult`
4. Register in `ToolRegistry._register_defaults()` or via entry point
5. Add tests in `tests/test_tools.py`

## Adding a new provider
1. Subclass `BaseProvider` in `sumospace/providers.py`
2. Implement `initialize()`, `complete()`, `stream()`
3. Add to `PROVIDERS` dict
4. Add optional dependency to `pyproject.toml`
5. Document in `docs/guides/providers.md`
```

---

## 10. Production & Deployment

### 10.1 — FastAPI Server Mode

```python
# sumospace/server.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SumoSpace API", version="0.1.0")

class RunRequest(BaseModel):
    task: str
    user_id: str = ""
    session_id: str = ""
    provider: str = "hf"
    dry_run: bool = False

class RunResponse(BaseModel):
    session_id: str
    success: bool
    final_answer: str
    intent: str
    steps_executed: int
    duration_ms: float
    error: str = ""

@app.post("/run", response_model=RunResponse)
async def run_task(request: RunRequest):
    from sumospace.kernel import KernelConfig, SumoKernel
    config = KernelConfig(
        provider=request.provider,
        user_id=request.user_id,
        session_id=request.session_id,
        scope_level="user" if request.user_id else "none",
        dry_run=request.dry_run,
        verbose=False,
    )
    async with SumoKernel(config) as kernel:
        trace = await kernel.run(request.task)
    return RunResponse(
        session_id=trace.session_id,
        success=trace.success,
        final_answer=trace.final_answer,
        intent=trace.intent.value,
        steps_executed=len(trace.step_traces),
        duration_ms=trace.duration_ms,
        error=trace.error,
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Launch:
```bash
uvicorn sumospace.server:app --host 0.0.0.0 --port 8080 --workers 4
```

### 10.2 — Docker Compose for Full Stack

```yaml
# docker-compose.yml
services:
  sumospace:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SUMO_PROVIDER=vllm
      - SUMO_VLLM_BASE_URL=http://vllm:8000
      - SUMO_SHELL_SANDBOX=true
      - SUMO_AUDIT_LOG_ENABLED=true
    volumes:
      - ./workspace:/workspace
      - sumo_db:/app/.sumo_db
    depends_on:
      - vllm

  vllm:
    image: vllm/vllm-openai:latest
    command: ["--model", "microsoft/Phi-3-mini-4k-instruct", "--dtype", "auto"]
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - hf_cache:/root/.cache/huggingface

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "4317:4317"

volumes:
  sumo_db:
  hf_cache:
```

---

## 11. Implementation Priority Order

Work through this in order. Each phase is releasable.

```
Phase 1 — Trust (Week 1-2)          Fixes 1.1, 1.2, 2.1-2.5, 3.1-3.5, 4.1-4.3
  Goal: No silent failures. No security holes. Correct behavior guaranteed.
  Release: v0.1.1 "Reliability patch"

Phase 2 — Customizability (Week 3-4) Section 5.1-5.5
  Goal: SumoSettings, plugin system, prompt templates, hooks, custom agents.
  Release: v0.2.0 "Fully configurable"

Phase 3 — Features (Week 5-6)        Section 6.1-6.8
  Goal: vLLM, streaming, plan cache, incremental ingest, audit log, watch.
  Release: v0.3.0 "Production features"

Phase 4 — Observability (Week 7)     Section 7.1-7.2
  Goal: OpenTelemetry, sumo logs, structured tracing.
  Release: v0.3.1

Phase 5 — Quality (Week 8)           Section 8.1-8.3
  Goal: 80%+ coverage, committee integration tests, scope race condition tests.
  Release: v0.4.0

Phase 6 — Documentation (Week 9)     Section 9.1-9.3
  Goal: Full MkDocs site, API reference, CHANGELOG, CONTRIBUTING.
  Release: v0.4.1

Phase 7 — Production (Week 10)       Section 10.1-10.2
  Goal: FastAPI server, Docker Compose, deployment guide.
  Release: v1.0.0 "Production ready"
```

---

## 12. Score Progression

| Phase | Score | What changes |
|---|---|---|
| Current | 52/100 | Baseline |
| After Phase 1 | 65/100 | Critical bugs fixed, security patched, reliable |
| After Phase 2 | 75/100 | Fully customizable, plugin system live |
| After Phase 3 | 82/100 | vLLM, real streaming, production features |
| After Phase 4 | 86/100 | Observable, debuggable |
| After Phase 5 | 91/100 | 80%+ test coverage, race conditions fixed |
| After Phase 6 | 95/100 | Full docs, DX polished |
| After Phase 7 | 100/100 | Production-deployable, industry-adoptable |

---

*Generated against SumoSpace commit `main` (2 commits). All code snippets are drop-in ready unless marked otherwise.*
