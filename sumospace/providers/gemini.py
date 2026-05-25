import os
from typing import AsyncIterator
from .base import BaseProvider, ProviderCapabilities
from sumospace.exceptions import ProviderNotConfiguredError

class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None, **kwargs):
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=False,
            preferred_fallback="legacy_json"
        )
        self.model = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = None

    async def initialize(self):
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "Gemini requires GOOGLE_API_KEY.\n"
                "Get one free at: https://aistudio.google.com/apikey\n"
                "Then set: export GOOGLE_API_KEY=your_key"
            )
        try:
            from google import genai
        except ImportError:
            raise ProviderNotConfiguredError(
                "Gemini package not installed. Run: pip install sumospace[gemini]"
            )
        self._client = genai.Client(api_key=self._api_key)

    async def complete(
        self, user: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048
    ) -> str:
        from google.genai import types
        prompt = f"{system}\n\n{user}" if system else user
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text

    async def complete_structured(
        self, user: str, system: str = "", schema: dict | None = None, temperature: float = 0.1, max_tokens: int = 2048
    ) -> str:
        from google.genai import types
        
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
            
        prompt = f"{system}\n\n{user}" if system else user
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs)
        )
        return response.text

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        from google.genai import types
        prompt = f"{system}\n\n{user}" if system else user
        response_stream = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        async for chunk in response_stream:
            yield chunk.text
