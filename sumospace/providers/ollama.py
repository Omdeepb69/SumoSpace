import asyncio
from typing import AsyncIterator
import json
import ollama

from .base import BaseProvider, ProviderCapabilities
from sumospace.exceptions import ProviderError, ProviderNotConfiguredError

OLLAMA_DEFAULT_MODELS = {
    "default": "phi3:mini",
    "fast":    "tinyllama",
    "capable": "mistral",
    "code":    "qwen2.5-coder",
}

class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = "default",
        base_url: str = "http://localhost:11434",
        auto_pull: bool = True,
        **kwargs,
    ):
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=True,  # enabled for modern ollama
            preferred_fallback="legacy_json"
        )
        self.model = OLLAMA_DEFAULT_MODELS.get(model, model)
        self.base_url = base_url
        self.auto_pull = auto_pull
        self._client = ollama.AsyncClient(host=self.base_url)

    async def initialize(self):
        from rich.console import Console
        console = Console()

        try:
            r = await self._client.list()
            available = [m.model for m in r.models]
            if self.model not in available and self.auto_pull:
                console.print(f"[dim]Pulling [cyan]{self.model}[/cyan] via Ollama...[/dim]")
                await self._pull_model()
        except Exception as e:
            raise ProviderNotConfiguredError(
                f"Ollama not running at {self.base_url} or error occurred.\n"
                f"Details: {e}\n"
                "Install from https://ollama.com, then run: ollama serve"
            )

    async def _pull_model(self):
        await self._client.pull(self.model)

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

        try:
            resp = await self._client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            return resp.get("message", {}).get("content", "")
        except Exception as e:
            raise ProviderError(f"Ollama API error: {e}") from e

    async def complete_structured(
        self,
        user: str,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            }
            if schema:
                kwargs["format"] = schema
                
            resp = await self._client.chat(**kwargs)
            return resp.get("message", {}).get("content", "")
        except Exception as e:
            raise ProviderError(f"Ollama API error: {e}") from e

    def format_tool_result(self, tool_call_id, tool_name, content) -> dict:
        return {
            "role": "tool",
            "content": str(content)
        }

    async def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        try:
            response = await self._client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                options={"temperature": 0.1}
            )
            
            if response.get("message", {}).get("tool_calls"):
                tool_calls = response["message"]["tool_calls"]
                return {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": tc["function"]["name"],
                            "name": tc["function"]["name"],
                            "arguments": dict(tc["function"]["arguments"])
                        }
                        for tc in tool_calls
                    ],
                    "assistant_message": response["message"]
                }
            else:
                # Modern fallback: structured JSON output, not text parsing
                forced_response = await self._client.chat(
                    model=self.model,
                    messages=messages + [{
                        "role": "user",
                        "content": "You must respond with a JSON object specifying which tool to call and its arguments."
                    }],
                    format={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"}
                        },
                        "required": ["name", "arguments"]
                    },
                    options={"temperature": 0.0}
                )
                
                content = forced_response.get("message", {}).get("content", "")
                try:
                    tool_call = json.loads(content)
                    return {
                        "type": "tool_calls",
                        "tool_calls": [{
                            "id": tool_call.get("name"),
                            "name": tool_call.get("name"),
                            "arguments": tool_call.get("arguments", {})
                        }],
                        "assistant_message": {"role": "assistant", "content": content}
                    }
                except json.JSONDecodeError:
                    return {
                        "type": "text",
                        "content": response.get("message", {}).get("content", ""),
                        "assistant_message": response.get("message", {})
                    }
        except Exception as e:
            raise ProviderError(f"Ollama native API error: {e}") from e

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        async for chunk in await self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": temperature}
        ):
            if token := chunk.get("message", {}).get("content"):
                yield token
