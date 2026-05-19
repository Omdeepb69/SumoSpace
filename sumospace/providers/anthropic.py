import os
from typing import AsyncIterator
from .base import BaseProvider, ProviderCapabilities
from sumospace.exceptions import ProviderNotConfiguredError

class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str | None = None, **kwargs):
        self.capabilities = ProviderCapabilities(
            structured_output=False,
            tool_calling=False,
            preferred_fallback="xml"
        )
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
