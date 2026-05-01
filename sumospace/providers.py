# sumospace/providers.py

"""
Provider Abstraction Layer
===========================
Default: HuggingFace local inference (Phi-3-mini) — zero API keys required.
Ollama: Zero API keys, requires Ollama server running locally.
Cloud providers (Gemini, OpenAI, Anthropic): Opt-in, require API keys + pip extras.

Priority for "auto" mode:
  1. Ollama (if server detected at localhost:11434)
  2. HuggingFace transformers (always available after pip install sumospace)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

from sumospace.exceptions import ProviderNotConfiguredError


# ─── Default Model Catalogues ─────────────────────────────────────────────────
# All models here are free, local, no API key.

HF_DEFAULT_MODELS = {
    "default":   "microsoft/Phi-3-mini-4k-instruct",   # 3.8B, best default balance
    "fast":      "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # 1.1B, CPU-friendly
    "capable":   "mistralai/Mistral-7B-Instruct-v0.3",  # 7B, needs ~8GB VRAM
    "code":      "Qwen/Qwen2.5-Coder-3B-Instruct",      # 3B, best for code tasks
    "reasoning": "microsoft/Phi-3-medium-4k-instruct",  # 14B, needs ~16GB VRAM
}

OLLAMA_DEFAULT_MODELS = {
    "default": "phi3:mini",      # ~2GB download, CPU runs fine
    "fast":    "tinyllama",      # ~600MB
    "capable": "mistral",        # ~4GB
    "code":    "qwen2.5-coder",  # ~2GB
}


# ─── Base ─────────────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str: ...

    async def stream(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        # Default: yield full response
        result = await self.complete(user, system, temperature)
        yield result

    async def initialize(self): pass


# ─── HuggingFace (DEFAULT — zero API key) ─────────────────────────────────────

class HuggingFaceProvider(BaseProvider):
    """
    Local HuggingFace inference via transformers pipeline.

    DEFAULT provider. Zero API keys. Works fully offline after first model download.
    Models are cached in ~/.cache/huggingface after first use.

    Args:
        model:        HuggingFace model ID or alias from HF_DEFAULT_MODELS.
        load_in_4bit: Use bitsandbytes 4-bit quantization (requires sumospace[local]).
                      Reduces VRAM from ~8GB to ~4GB for 7B models.
        device:       "auto", "cpu", "cuda", "mps" (Apple Silicon).
    """
    name = "hf"

    def __init__(
        self,
        model: str = "default",
        load_in_4bit: bool = False,
        device: str = "auto",
    ):
        self.model_id = HF_DEFAULT_MODELS.get(model, model)
        self.load_in_4bit = load_in_4bit
        self.device = device
        self._pipe = None

    async def initialize(self):
        import asyncio
        from rich.console import Console
        console = Console()
        console.print(
            f"[dim]Loading [cyan]{self.model_id}[/cyan] locally "
            f"(first run downloads ~2-4GB, cached after)...[/dim]"
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        import torch
        from transformers import pipeline, BitsAndBytesConfig

        bnb_config = None
        if self.load_in_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                pass  # bitsandbytes not installed — full precision fallback

        if self.device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_map = "mps"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
        else:
            device_map = self.device
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16

        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

    async def complete(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        import asyncio

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
                return_full_text=False,
            ),
        )

        generated = result[0].get("generated_text", "")
        if isinstance(generated, list):
            generated = generated[-1].get("content", "") if generated else ""
        return generated.strip()

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        result = await self.complete(user, system, temperature)
        yield result


# ─── Ollama (zero API key, requires local Ollama server) ─────────────────────

class OllamaProvider(BaseProvider):
    """
    Ollama local inference. Zero API keys.
    Requires: Ollama installed + running (https://ollama.com — free).

    Args:
        model:     Ollama model tag or alias from OLLAMA_DEFAULT_MODELS.
        base_url:  Ollama server URL (default: http://localhost:11434)
        auto_pull: If True, automatically pull model if not available.
    """
    name = "ollama"

    def __init__(
        self,
        model: str = "default",
        base_url: str = "http://localhost:11434",
        auto_pull: bool = True,
        **kwargs,
    ):
        self.model = OLLAMA_DEFAULT_MODELS.get(model, model)
        self.base_url = base_url
        self.auto_pull = auto_pull
        self._client = None

    async def initialize(self):
        import httpx
        from rich.console import Console
        console = Console()

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=300)

        try:
            r = await self._client.get("/api/tags")
            available = [m["name"] for m in r.json().get("models", [])]
            if self.model not in available and self.auto_pull:
                console.print(f"[dim]Pulling [cyan]{self.model}[/cyan] via Ollama...[/dim]")
                await self._pull_model()
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Ollama not running at {self.base_url}.\n"
                "Install from https://ollama.com, then run: ollama serve"
            )

    async def _pull_model(self):
        async with self._client.stream(
            "POST", "/api/pull", json={"name": self.model}
        ) as r:
            async for _ in r.aiter_lines():
                pass  # Stream pull progress silently

    async def complete(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = await self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        return resp.json()["message"]["content"]

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        import json

        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        async with self._client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
        ) as r:
            async for line in r.aiter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("message", {}).get("content"):
                        yield token


# ─── Cloud providers (opt-in, require API keys + pip extras) ─────────────────

class GeminiProvider(BaseProvider):
    """
    Google Gemini API.
    Requires: pip install sumospace[gemini] + GOOGLE_API_KEY env var.
    """
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None, **kwargs):
        self.model = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    async def initialize(self):
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "Gemini requires GOOGLE_API_KEY.\n"
                "Get one free at: https://aistudio.google.com/apikey\n"
                "Then set: export GOOGLE_API_KEY=your_key"
            )
        try:
            import google.generativeai as genai
        except ImportError:
            raise ProviderNotConfiguredError(
                "Gemini package not installed. Run: pip install sumospace[gemini]"
            )
        genai.configure(api_key=self._api_key)

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
            ),
        )
        prompt = f"{system}\n\n{user}" if system else user
        response = await model.generate_content_async(prompt)
        return response.text

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
        prompt = f"{system}\n\n{user}" if system else user
        response = await model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            yield chunk.text


class OpenAIProvider(BaseProvider):
    """
    OpenAI API.
    Requires: pip install sumospace[openai] + OPENAI_API_KEY env var.
    """
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, **kwargs):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def initialize(self):
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "OpenAI requires OPENAI_API_KEY.\n"
                "Get one at: https://platform.openai.com/api-keys\n"
                "Then set: export OPENAI_API_KEY=your_key"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ProviderNotConfiguredError(
                "OpenAI package not installed. Run: pip install sumospace[openai]"
            )
        self._client = AsyncOpenAI(api_key=self._api_key)

    async def complete(
        self, user: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        async for chunk in await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=True
        ):
            if delta := chunk.choices[0].delta.content:
                yield delta


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude API.
    Requires: pip install sumospace[anthropic] + ANTHROPIC_API_KEY env var.
    """
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str | None = None, **kwargs):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def initialize(self):
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "Anthropic requires ANTHROPIC_API_KEY.\n"
                "Get one at: https://console.anthropic.com\n"
                "Then set: export ANTHROPIC_API_KEY=your_key"
            )
        try:
            import anthropic
        except ImportError:
            raise ProviderNotConfiguredError(
                "Anthropic package not installed. Run: pip install sumospace[anthropic]"
            )
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    async def complete(
        self, user: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as s:
            async for text in s.text_stream:
                yield text


# ─── Provider Registry ────────────────────────────────────────────────────────

PROVIDERS: dict[str, type[BaseProvider]] = {
    "hf":          HuggingFaceProvider,
    "huggingface": HuggingFaceProvider,
    "ollama":      OllamaProvider,
    "gemini":      GeminiProvider,
    "openai":      OpenAIProvider,
    "anthropic":   AnthropicProvider,
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

    Usage:
        # Default — no API key needed
        router = ProviderRouter()
        await router.initialize()

        # Specific local model
        router = ProviderRouter(provider="ollama", model="mistral")

        # Cloud (needs API key + pip extra)
        router = ProviderRouter(provider="gemini", model="gemini-1.5-pro")

    Special values for provider:
        "auto"      — Tries Ollama first, falls back to HuggingFace
        "hf"        — HuggingFace transformers (default)
        "ollama"    — Ollama local server
        "gemini" / "openai" / "anthropic" — Cloud (requires key + pip extra)
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

    async def complete(self, **kwargs) -> str:
        return await self._provider.complete(**kwargs)

    async def stream(self, **kwargs) -> AsyncIterator[str]:
        return self._provider.stream(**kwargs)

    @property
    def provider_name(self) -> str:
        return self._provider_name
