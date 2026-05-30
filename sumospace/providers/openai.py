import os
from typing import AsyncIterator
from .base import BaseProvider, ProviderCapabilities
from sumospace.exceptions import ProviderNotConfiguredError

class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, **kwargs):
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=False,
            preferred_fallback="legacy_json"
        )
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

    async def complete_structured(
        self, user: str, system: str = "", schema: dict | None = None, temperature: float = 0.1, max_tokens: int = 2048
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True
                }
            }
            
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "tools": tools,
            "tool_choice": "auto",
        }
            
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        
        if choice.message.tool_calls:
            tool_calls = []
            import json
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "assistant_message": choice.message.model_dump()
            }
        else:
            return {"type": "text", "content": choice.message.content or ""}

    def format_tool_result(self, tool_call_id: str, name: str, content: str) -> dict | list[dict]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        }

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
