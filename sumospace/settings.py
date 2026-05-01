from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
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
