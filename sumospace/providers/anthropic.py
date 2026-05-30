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

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"]
            })
            
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.1,
            "messages": messages,
            "tools": anthropic_tools,
        }
        
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        if system:
            kwargs["system"] = system
            kwargs["messages"] = [m for m in messages if m["role"] != "system"]

        response = await self._client.messages.create(**kwargs)
        
        tool_calls = []
        text_content = ""
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
            elif block.type == "text":
                text_content += block.text
                
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "assistant_message": {"role": "assistant", "content": [b.model_dump() for b in response.content]}
            }
        return {"type": "text", "content": text_content}

    def format_tool_result(self, tool_call_id: str, name: str, content: str) -> dict | list[dict]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content
                }
            ]
        }

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
