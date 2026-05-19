from typing import AsyncIterator
import json
from .base import BaseProvider, ProviderCapabilities
from sumospace.exceptions import ProviderNotConfiguredError

class VLLMProvider(BaseProvider):
    name = "vllm"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        max_concurrent: int = 10,
        **kwargs
    ):
        # vLLM supports guided decoding natively with JSON schema.
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=False,
            preferred_fallback="legacy_json"
        )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None

    async def initialize(self):
        import httpx
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300,
        )
        try:
            r = await self._client.get("/health")
            r.raise_for_status()
        except Exception:
            raise ProviderNotConfiguredError(
                f"vLLM server not reachable at {self.base_url}.\n"
                f"Start with: vllm serve {self.model} --dtype auto"
            )

    async def complete(
        self, user: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def complete_structured(
        self, user: str, system: str = "", schema: dict | None = None, temperature: float = 0.1, max_tokens: int = 2048
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True
                }
            }

        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            },
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if delta := data["choices"][0]["delta"].get("content"):
                        yield delta
