# sumospace/config.py

"""
Global configuration.
All fields optional — nothing required for the default local setup.
Cloud API keys are only read when the corresponding provider is explicitly selected.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field


class SumoConfig(BaseSettings):
    """
    Sumospace global config. Loaded from environment or .env file.
    NONE of these are required for the default local setup.
    """

    # ── Cloud provider keys (all optional) ────────────────────────────────────
    google_api_key: str = Field(default="", description="For Gemini + Google Embeddings")
    openai_api_key: str = Field(default="", description="For OpenAI GPT models")
    anthropic_api_key: str = Field(default="", description="For Claude models")

    # ── Local provider config (no keys needed) ────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    hf_default_model: str = "microsoft/Phi-3-mini-4k-instruct"
    hf_load_in_4bit: bool = False                # Set True to halve VRAM usage
    hf_cache_dir: str = ""                       # Empty = use HF default ~/.cache/huggingface

    # ── Default provider selection ─────────────────────────────────────────────
    default_provider: str = "hf"                 # hf | ollama | auto
    default_model: str = "default"
    default_embedding: str = "local"             # local | google | openai
    default_embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ── Storage ────────────────────────────────────────────────────────────────
    chroma_path: str = ".sumo_db"
    workspace: str = "."

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "SUMO_",   # All env vars prefixed: SUMO_DEFAULT_PROVIDER=ollama
        "extra": "ignore",
    }


# Singleton config instance
config = SumoConfig()
