import os
from typing import AsyncIterator
from .base import BaseProvider, ProviderCapabilities
from sumospace.exceptions import ProviderNotConfiguredError

class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None, **kwargs):
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=True,
            preferred_fallback=None
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

    def format_tool_result(self, tool_call_id, tool_name, content) -> dict:
        return {
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": tool_name,
                    "response": {"result": str(content)}
                }
            }]
        }

    def _convert_messages(self, messages: list[dict]) -> list:
        gemini_messages = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "parts"):
                gemini_messages.append(m)
                continue
                
            role = m.get("role", "user")
            
            if role == "system":
                gemini_messages.append({"role": "user", "parts": [{"text": m.get("content", "")}]})
            elif role == "user":
                if "parts" in m:
                    gemini_messages.append(m)
                else:
                    gemini_messages.append({"role": "user", "parts": [{"text": m.get("content", "")}]})
            elif role == "assistant":
                if "parts" in m:
                    m_copy = dict(m)
                    m_copy["role"] = "model"
                    gemini_messages.append(m_copy)
                else:
                    gemini_messages.append({"role": "model", "parts": [{"text": m.get("content", "")}]})
            elif role == "tool":
                if "parts" in m:
                    m_copy = dict(m)
                    m_copy["role"] = "user"
                    gemini_messages.append(m_copy)
                else:
                    gemini_messages.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": m.get("name", "tool"),
                                "response": {"result": str(m.get("content", ""))}
                            }
                        }]
                    })
        return gemini_messages

    async def complete_with_tools(self, messages, tools) -> dict:
        from google import genai
        declarations = []
        for t in tools:
            fn = t["function"]
            declarations.append(
                genai.types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {})
                )
            )
        gemini_tools = [genai.types.Tool(function_declarations=declarations)]
        
        gemini_messages = self._convert_messages(messages)
        
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=genai.types.GenerateContentConfig(tools=gemini_tools)
        )
        
        if not response.candidates:
            return {"type": "text", "content": ""}
            
        candidate = response.candidates[0].content
        tool_calls = []
        text_parts = []
        
        if not hasattr(candidate, "parts") or not candidate.parts:
            return {"type": "text", "content": ""}
            
        for part in candidate.parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append({
                    "id": part.function_call.name,
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args)
                })
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)
        
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "assistant_message": candidate
            }
        else:
            return {
                "type": "text",
                "content": "".join(text_parts),
                "assistant_message": candidate
            }
