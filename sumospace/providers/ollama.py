from typing import AsyncIterator
import json
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
            tool_calling=False,
            preferred_fallback="legacy_json"
        )
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
                pass

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

        import httpx
        
        try:
            resp = await self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama returned HTTP {e.response.status_code}. "
                f"Is the model '{self.model}' loaded? Run: ollama pull {self.model}"
            ) from e
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot reach Ollama at {self.base_url}. Run: ollama serve"
            )

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

        import httpx
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if schema:
                payload["format"] = schema

            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama returned HTTP {e.response.status_code}. "
                f"Is the model '{self.model}' loaded? Run: ollama pull {self.model}"
            ) from e
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot reach Ollama at {self.base_url}. Run: ollama serve"
            )

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        import httpx
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "tools": tools,
                "options": {"temperature": 0.1, "num_predict": 4096},
            }

            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = []
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    })
                return {
                    "type": "tool_calls",
                    "tool_calls": tool_calls,
                    "assistant_message": message,  # raw Ollama assistant message with tool_calls
                }
            else:
                return {"type": "text", "content": message.get("content", "")}
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama returned HTTP {e.response.status_code}. "
                f"Is the model '{self.model}' loaded? Run: ollama pull {self.model}"
            ) from e
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot reach Ollama at {self.base_url}. Run: ollama serve"
            )

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
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
