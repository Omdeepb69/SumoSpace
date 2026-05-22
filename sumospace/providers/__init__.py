import os
from typing import AsyncIterator
from sumospace.exceptions import ProviderError, ProviderNotConfiguredError
from .base import BaseProvider, ProviderCapabilities

from .hf import HuggingFaceProvider, HF_DEFAULT_MODELS
from .ollama import OllamaProvider, OLLAMA_DEFAULT_MODELS
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .vllm import VLLMProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "hf":          HuggingFaceProvider,
    "huggingface": HuggingFaceProvider,
    "ollama":      OllamaProvider,
    "gemini":      GeminiProvider,
    "openai":      OpenAIProvider,
    "anthropic":   AnthropicProvider,
    "vllm":        VLLMProvider,
}

async def _detect_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running at base_url."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2) as client:
            await client.get(f"{base_url}/api/tags")
        return True
    except Exception:
        return False


class ProviderRouter:
    """
    Factory + proxy for all providers.
    """

    def __init__(
        self,
        provider: str = "hf",
        model: str | None = None,
        **kwargs,
    ):
        self._provider_name = provider
        self._model = model
        self._kwargs = kwargs
        self._provider: BaseProvider | None = None
        self._secondary: BaseProvider | None = None

    async def initialize(self):
        from rich.console import Console
        console = Console()

        if self._provider_name == "auto":
            if await _detect_ollama():
                console.print("[dim]Auto-detected Ollama — using ollama/phi3:mini[/dim]")
                self._provider_name = "ollama"
                self._model = self._model or "default"
            else:
                console.print("[dim]Ollama not detected — using HuggingFace/Phi-3-mini[/dim]")
                self._provider_name = "hf"
                self._model = self._model or "default"

        if self._provider_name not in PROVIDERS:
            raise ProviderNotConfiguredError(
                f"Unknown provider '{self._provider_name}'. "
                f"Available: {list(PROVIDERS.keys())}"
            )

        cls = PROVIDERS[self._provider_name]
        init_kwargs = self._kwargs.copy()
        if self._model:
            init_kwargs["model"] = self._model

        self._provider = cls(**init_kwargs)
        await self._provider.initialize()
        
        # If cloud provider, set HF as secondary fallback (DISABLED for Kaggle to save RAM)
        # if self._provider_name in ["gemini", "openai", "anthropic"]:
        #     self._secondary = HuggingFaceProvider(model="default")
        #     await self._secondary.initialize()

    def get_output_strategy(self) -> str:
        """Determines the best strategy for structured output routing."""
        caps = self._provider.capabilities
        if caps.structured_output:
            return "structured"
        if caps.tool_calling:
            return "tool_calling"
        if caps.preferred_fallback == "xml":
            return "xml"
        return "legacy_json"

    async def complete_structured(
        self,
        user: str,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        import httpx
        import json
        import time
        from pathlib import Path
        from rich.console import Console

        strategy = self.get_output_strategy()
        
        # ── Raw structured output logging ──
        log_dir = Path("/tmp/sumospace_logs/structured_output_raw")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp": time.time(),
            "provider": self._provider_name,
            "strategy": strategy,
            "schema_title": (schema or {}).get("title", "unknown"),
            "user_prompt_preview": user[:200],
        }

        try:
            if strategy == "structured" or strategy == "tool_calling":
                res = await self._provider.complete_structured(
                    user=user, system=system, schema=schema, 
                    temperature=temperature, max_tokens=max_tokens
                )
            elif strategy == "xml":
                res = await self._provider._complete_xml(
                    user=user, system=system, schema=schema, 
                    temperature=temperature, max_tokens=max_tokens
                )
            else:
                # legacy_json strategy bypasses structured outputs completely
                # and just relies on standard complete. The caller must parse it.
                res = await self._provider.complete(
                    user=user, system=system, temperature=temperature, max_tokens=max_tokens
                )
                
            if not res or not res.strip():
                raise ProviderError("Provider returned empty response")

            # Log success
            log_entry["status"] = "success"
            log_entry["raw_output_preview"] = res[:500]
            log_entry["raw_output_length"] = len(res)
            try:
                log_file = log_dir / f"{int(time.time()*1000)}.json"
                log_file.write_text(json.dumps(log_entry, indent=2))
            except Exception:
                pass

            return res
        except (httpx.ConnectError, httpx.TimeoutException, ProviderError, ProviderNotConfiguredError, ValueError) as e:
            # Log failure
            log_entry["status"] = "error"
            log_entry["error"] = str(e)
            try:
                log_file = log_dir / f"{int(time.time()*1000)}_error.json"
                log_file.write_text(json.dumps(log_entry, indent=2))
            except Exception:
                pass

            if hasattr(e, "response") and getattr(e, "response", None) and e.response.status_code in [400, 401]:
                raise
            
            if self._secondary:
                Console().print(f"[yellow]Primary provider failed structured generation ({e}), falling back to {self._secondary.name}...[/yellow]")
                
                # Check secondary strategy
                sec_caps = self._secondary.capabilities
                if sec_caps.structured_output:
                    return await self._secondary.complete_structured(user, system, schema, temperature, max_tokens)
                elif sec_caps.preferred_fallback == "xml":
                    return await self._secondary._complete_xml(user, system, schema, temperature, max_tokens)
                else:
                    return await self._secondary.complete(user, system, temperature, max_tokens)
            raise

    async def complete(self, **kwargs) -> str:
        import httpx
        try:
            res = await self._provider.complete(**kwargs)
            if not res or not res.strip():
                raise ProviderError("Provider returned empty response")
            return res
        except (httpx.ConnectError, httpx.TimeoutException, ProviderError, ProviderNotConfiguredError) as e:
            if hasattr(e, "response") and getattr(e, "response", None) and e.response.status_code in [400, 401]:
                raise
            
            if self._secondary:
                from rich.console import Console
                Console().print(f"[yellow]Primary provider failed ({e}), falling back to {self._secondary.name}...[/yellow]")
                res = await self._secondary.complete(**kwargs)
                if not res or not res.strip():
                    raise ProviderError("Fallback provider also returned empty response")
                return res
            raise

    async def stream(self, **kwargs) -> AsyncIterator[str]:
        import httpx
        try:
            any_tokens = False
            async for chunk in self._provider.stream(**kwargs):
                any_tokens = True
                yield chunk
            
            if not any_tokens:
                raise ProviderError("Provider returned empty stream")
                
        except (httpx.ConnectError, httpx.TimeoutException, ProviderError, ProviderNotConfiguredError) as e:
            if hasattr(e, "response") and getattr(e, "response", None) and e.response.status_code in [400, 401]:
                raise
                
            if self._secondary:
                from rich.console import Console
                Console().print(f"[yellow]Primary provider failed ({e}), falling back to {self._secondary.name}...[/yellow]")
                async for chunk in self._secondary.stream(**kwargs):
                    yield chunk
                return
            raise

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_instance(self) -> BaseProvider:
        return self._provider
