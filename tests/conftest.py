# tests/conftest.py

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# ─── Mock provider (no actual model needed for tests) ─────────────────────────

class MockProvider:
    """Fake provider that returns deterministic responses."""
    name = "mock"

    def __init__(self):
        self._react_call_count = 0

    async def complete(self, user: str, system: str = "", temperature: float = 0.2,
                       max_tokens: int = 2048) -> str:
        # Return structured JSON for classifier/committee calls
        if "intent" in system.lower() or "classification" in system.lower():
            return '{"intent": "general_qa", "confidence": 0.8, "needs_execution": false, "needs_web": false, "needs_retrieval": false, "reasoning": "mock"}'
        if "steps" in system.lower() or "planner" in system.lower():
            return '{"reasoning": "mock plan", "estimated_duration_s": 5, "steps": [{"step_number": 1, "tool": "shell", "description": "echo test", "parameters": {"command": "echo hello"}, "expected_output": "hello", "critical": false}]}'
        if "risks" in system.lower() or "critic" in system.lower():
            return '{"risks": [], "blockers": [], "suggestions": [], "verdict": "approve", "verdict_reason": "looks good"}'
        if "approved" in system.lower() or "resolver" in system.lower():
            return '{"approved": true, "approval_notes": "approved", "reasoning": "ok", "estimated_duration_s": 5, "steps": [{"step_number": 1, "tool": "shell", "description": "echo", "parameters": {"command": "echo hello"}, "critical": false}]}'
        if "reflect" in system.lower():
            return '{"success": true, "reason": "mock reflection", "retry": false}'
        if "summarize" in system.lower() or "summarizer" in system.lower():
            return "Summary of previous steps."
        return f"Mock response to: {user[:50]}"

    async def complete_structured(self, user: str = "", system: str = "", schema: dict | None = None,
                                   temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """Delegates to complete() — the mock already returns valid JSON."""
        return await self.complete(user=user, system=system, temperature=temperature, max_tokens=max_tokens)

    async def complete_with_tools(self, messages: list[dict], tools: list[dict], **kwargs) -> dict:
        system = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""

        # Detect ReAct execution context (autonomous agent with real tools like shell, read_file, etc.)
        is_react = "autonomous" in system.lower() or "use the provided tools" in system.lower()

        if is_react:
            self._react_call_count += 1
            # First call: execute a shell tool to satisfy MIN_STEPS_BEFORE_FINISH
            if self._react_call_count == 1:
                # Find a shell tool in the schema, or use the first available tool
                tool_name = "shell"
                for t in (tools or []):
                    if t.get("function", {}).get("name") == "shell":
                        tool_name = "shell"
                        break
                else:
                    if tools:
                        tool_name = tools[0]["function"]["name"]
                return {
                    "type": "tool_calls",
                    "tool_calls": [{
                        "id": "mock_react_id",
                        "name": tool_name,
                        "arguments": {"command": "echo done"}
                    }]
                }
            # Subsequent calls: signal completion
            else:
                self._react_call_count = 0  # Reset for next test
                return {"type": "text", "content": "Task completed successfully."}

        raw = await self.complete(user="", system=system)
        import json
        try:
            args = json.loads(raw)
        except Exception:
            # For fallback tests returning bad json
            return {"type": "text", "content": raw}

        # Derive the expected tool name directly from the tools schema (most reliable signal)
        if tools and tools[0].get("function", {}).get("name"):
            tool_name = tools[0]["function"]["name"]
        else:
            # Last-resort fallback: system prompt heuristics
            tool_name = "submit_plan"
            if "critic" in system.lower():
                tool_name = "submit_critique"
            elif "resolver" in system.lower():
                tool_name = "submit_resolution"
            
        return {
            "type": "tool_calls",
            "tool_calls": [{
                "id": "mock_id",
                "name": tool_name,
                "arguments": args
            }]
        }

    def format_tool_result(self, tool_call_id: str, name: str, content: str) -> dict | list[dict]:
        """Format a tool result into a message for the conversation."""
        return {"role": "tool", "content": content, "name": name}

    async def stream(self, user: str, system: str = "", temperature: float = 0.2):
        yield await self.complete(user, system, temperature)

    async def initialize(self):
        pass


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with some sample files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        'def hello():\n    """Say hello."""\n    return "hello"\n'
    )
    (tmp_path / "src" / "utils.py").write_text(
        'import os\n\ndef get_cwd():\n    return os.getcwd()\n'
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        'from src.main import hello\n\ndef test_hello():\n    assert hello() == "hello"\n'
    )
    (tmp_path / "README.md").write_text("# Test Project\nA sample project for testing.\n")
    return tmp_path


@pytest.fixture
def tmp_chroma(tmp_path):
    """Temporary ChromaDB path."""
    return str(tmp_path / "test_db")
