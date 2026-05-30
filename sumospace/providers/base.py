from abc import ABC, abstractmethod
from typing import AsyncIterator, Literal, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class ProviderCapabilities:
    structured_output: bool = False
    tool_calling: bool = False
    preferred_fallback: Literal["xml", "legacy_json"] = "legacy_json"


class BaseProvider(ABC):
    name: str = "base"
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    async def complete(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str: ...

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        """
        Native tool calling endpoint. Must return a dict containing either:
        {"type": "text", "content": "..."}
        or
        {"type": "tool_calls", "tool_calls": [{\"name\": \"...\", \"arguments\": {...}}],
         "assistant_message": <dict to append to messages as the assistant turn>}

        The assistant_message is the raw provider message that should be appended
        to the conversation so the model can track what it asked for.
        """
        raise NotImplementedError(f"Native tool calling not supported by {self.name}")

    def format_tool_result(self, tool_call_id: str, name: str, content: str) -> dict | list[dict]:
        """
        Format a tool execution result into the provider-specific message structure.
        """
        return {"role": "tool", "content": content, "name": name}

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

    async def complete_structured(
        self,
        user: str,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Returns output guaranteed to match schema when provider supports it.
        Falls back to complete() when schema support unavailable.
        """
        # If the subclass does not implement a native version, fallback to standard complete
        # Depending on strategy routing in ProviderRouter, this might be bypassed entirely.
        return await self.complete(user, system, temperature, max_tokens)

    async def _complete_xml(
        self,
        user: str,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Helper for providers preferring XML fallback.
        Wraps schema fields into a system prompt asking for XML tags,
        then parses the resulting XML and returns a JSON string.
        """
        import json
        
        xml_instructions = (
            "\n\nOUTPUT FORMAT REQUIRED:\n"
            "You must respond ONLY in XML format, wrapped in <response> tags. "
            "Do not include any other text.\n"
        )
        
        if schema:
            xml_instructions += "Your XML must contain the following tags corresponding to these fields:\n"
            for prop_name, prop_details in schema.get("properties", {}).items():
                desc = prop_details.get("description", "")
                xml_instructions += f"- <{prop_name}>: {desc}\n"
        
        augmented_system = (system + xml_instructions).strip()
        
        raw_response = await self.complete(user, augmented_system, temperature, max_tokens)
        
        try:
            # Try to extract <response>...</response>
            start_idx = raw_response.find("<response>")
            end_idx = raw_response.find("</response>")
            if start_idx != -1 and end_idx != -1:
                xml_content = raw_response[start_idx:end_idx + len("</response>")]
            else:
                # Wrap it if missing
                xml_content = f"<response>\n{raw_response}\n</response>"
                
            root = ET.fromstring(xml_content)
            
            extracted_data = {}
            for child in root:
                # Basic parsing: assume string or attempt JSON parse if it looks like a complex object
                # For more complex structures like lists of dicts (e.g. steps), it might be serialized as JSON string inside XML
                text = (child.text or "").strip()
                if text.startswith("[") or text.startswith("{"):
                    try:
                        extracted_data[child.tag] = json.loads(text)
                    except json.JSONDecodeError:
                        extracted_data[child.tag] = text
                elif text.lower() == "true":
                    extracted_data[child.tag] = True
                elif text.lower() == "false":
                    extracted_data[child.tag] = False
                elif text.isdigit():
                    extracted_data[child.tag] = int(text)
                else:
                    extracted_data[child.tag] = text
                    
            # Auto-inject protocol_version if missing and required by schema
            if schema and "protocol_version" in schema.get("properties", {}) and "protocol_version" not in extracted_data:
                extracted_data["protocol_version"] = schema["properties"]["protocol_version"].get("default", "1.0")

            return json.dumps(extracted_data)
        except Exception as e:
            # If XML parsing fails, raise so legacy fallback can take over
            raise ValueError(f"XML parsing failed: {e}\nRaw output: {raw_response}")
