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

    async def complete_structured(
        self, user: str, system: str = "", schema: dict | None = None, temperature: float = 0.1, max_tokens: int = 2048
    ) -> str:
        import google.generativeai as genai
        
        gen_config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        if schema:
            gen_config_kwargs["response_mime_type"] = "application/json"
            gen_config_kwargs["response_schema"] = schema
            
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(**gen_config_kwargs),
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
