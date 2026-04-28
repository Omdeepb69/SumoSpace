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
        return f"Mock response to: {user[:50]}"

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
