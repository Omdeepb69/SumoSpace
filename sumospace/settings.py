from typing import Optional, TYPE_CHECKING, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from sumospace.kernel import KernelConfig
class SumoSettings(BaseSettings):
    """
    Centralized configuration layer for SumoSpace.
    Values are populated in this order of precedence:
    1. Explicit keyword arguments on instantiation
    2. Environment variables (e.g. SUMO_PROVIDER)
    3. Values loaded from a .env file
    4. Default values defined here
    """
    model_config = SettingsConfigDict(
        env_prefix="SUMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    provider: str = "hf"
    model: str = "default"

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    require_consensus: bool = True
    
    # ── Inference Pipeline Toggles ───────────────────────────────────────────
    committee_enabled: bool = Field(
        True,
        description=(
            "If False, bypasses the three-agent committee entirely. "
            "The kernel sends the task directly to the provider as a single inference call. "
            "Faster and cheaper, but no planning, no safety review, no execution steps. "
            "Best for simple Q&A, summarisation, or chat use cases."
        )
    )
    committee_mode: Literal["full", "plan_only", "critique_only"] = Field(
        "full",
        description=(
            "Controls which committee agents run when committee_enabled=True.\n"
            "'full'          — Planner + Critic + Resolver (default, safest)\n"
            "'plan_only'     — Planner only, no critique or resolution. Faster.\n"
            "'critique_only' — Planner + Critic, no Resolver. Plan executes if Critic approves."
        )
    )
    committee_temperature: float = 0.1
    committee_max_tokens: int = 2048
    
    execution_enabled: bool = Field(
        True,
        description=(
            "If False, the kernel plans and deliberates but does not execute any tools. "
            "The plan is returned in the trace but no filesystem, shell, or web calls are made. "
            "Equivalent to dry_run=True but permanent and settings-driven."
        )
    )
    rag_enabled: bool = Field(
        True,
        description="If False, skips vector store retrieval entirely. Faster for tasks that don't need codebase context."
    )
    rag_top_k_final: int = 5
    memory_enabled: bool = Field(
        True,
        description="If False, disables episodic memory read and write. Each run is completely stateless."
    )
    shell_sandbox: bool = False
    
    max_retries: int = 3
    execution_timeout: int = 120
    verbose: bool = True
    dry_run: bool = False
    hf_load_in_4bit: bool = False

    secondary_provider: Optional[str] = None
    secondary_model: Optional[str] = None

    workspace: str = "."

    scope_level: str = "user"
    user_id: str = ""
    session_id: str = ""
    project_id: str = ""
    chroma_base: str = ".sumo_db"
    max_chunks_per_scope: Optional[int] = None

    # ── Prompt Templates ──────────────────────────────────────────────────────
    prompt_template_path: Optional[str] = None  # Directory containing custom prompt .txt files

    # ── Hooks ─────────────────────────────────────────────────────────────────
    auto_load_hooks: bool = False
    """
    If True, automatically load hooks from .sumo_hooks.py in the workspace.
    Only enable if you trust the contents of your workspace directory.
    """
    hooks_module: Optional[str] = None  # Path or dotted module to load hooks from

    # ── Observability ──────────────────────────────────────────────────────────
    telemetry_enabled: bool = False
    telemetry_endpoint: str = "http://localhost:4317"

    # ── Presets ───────────────────────────────────────────────────────────────
    
    @classmethod
    def for_chat(cls, **kwargs) -> "SumoSettings":
        """
        Conversational chat with memory but no planning or tool execution.
        Use this for multi-turn conversations where the model needs to
        remember what was said earlier in the session.
        No committee, no RAG, no tool calls. Fast response times.
        """
        defaults = {
            "committee_enabled": False,
            "rag_enabled": False,
            "memory_enabled": True,
            "execution_enabled": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_chat_with_context(cls, **kwargs) -> "SumoSettings":
        """
        Conversational chat grounded in your codebase or documents.
        Use this when users ask questions about ingested content —
        'explain this function', 'where is X defined', 'summarise this doc'.
        No committee, no tool execution. RAG + memory enabled.
        """
        defaults = {
            "committee_enabled": False,
            "rag_enabled": True,
            "memory_enabled": True,
            "execution_enabled": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_chat_stateless(cls, **kwargs) -> "SumoSettings":
        """
        Pure stateless single-turn inference. Fastest possible response.
        Every message is independent — no memory of previous turns.
        Use for one-shot Q&A, summarisation, or classification tasks
        where conversation history is irrelevant.
        """
        defaults = {
            "committee_enabled": False,
            "rag_enabled": False,
            "memory_enabled": False,
            "execution_enabled": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_coding(cls, **kwargs) -> "SumoSettings":
        """Full pipeline optimised for code tasks."""
        defaults = {
            "committee_enabled": True,
            "committee_mode": "full",
            "rag_enabled": True,
            "rag_top_k_final": 8,
            "shell_sandbox": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_research(cls, **kwargs) -> "SumoSettings":
        """Planning + web search, no code execution."""
        defaults = {
            "committee_enabled": True,
            "committee_mode": "plan_only",
            "rag_enabled": True,
            "execution_enabled": True,
            "shell_sandbox": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_review(cls, **kwargs) -> "SumoSettings":
        """Plan and critique only — never executes. Safe for untrusted tasks."""
        defaults = {
            "committee_enabled": True,
            "committee_mode": "full",
            "execution_enabled": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def from_file(cls, path: str) -> "SumoSettings":
        """Load settings explicitly from a specific file."""
        return cls(_env_file=path)

    def to_kernel_config(self) -> "KernelConfig":
        """
        Compatibility shim for code still using KernelConfig directly.
        Deprecated: Pass SumoSettings to SumoKernel directly instead.
        Will be removed in v1.0.
        """
        from sumospace.kernel import KernelConfig
        return KernelConfig(
            provider=self.provider,
            model=self.model,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            require_consensus=self.require_consensus,
            max_retries=self.max_retries,
            execution_timeout=self.execution_timeout,
            verbose=self.verbose,
            dry_run=self.dry_run,
            hf_load_in_4bit=self.hf_load_in_4bit,
            secondary_provider=self.secondary_provider,
            secondary_model=self.secondary_model,
            workspace=self.workspace,
            scope_level=self.scope_level,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            chroma_base=self.chroma_base,
            chroma_path=self.chroma_base,
            max_chunks_per_scope=self.max_chunks_per_scope,
        )
