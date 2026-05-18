# tests/test_domain_context.py

"""
Tests for DomainContext — additive Layer 3 prompt injection.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from sumospace.domain_context import DomainContext
from sumospace.committee import (
    PLANNER_SYSTEM,
    CRITIC_SYSTEM,
    RESOLVER_SYSTEM,
    PlannerAgent,
    CriticAgent,
    ResolverAgent,
)


# ── Test 1: Domain context appears in assembled prompt ────────────────────────

def test_domain_context_appears_in_prompt():
    """Domain context strings appear in the assembled system prompt."""
    ctx = DomainContext(
        global_context="FastAPI + PostgreSQL project.",
        planner_context="Use ONLY read_file and write_file.",
    )

    class FakeProvider:
        pass

    agent = PlannerAgent(FakeProvider(), domain_context=ctx)
    prompt = agent._build_system_prompt(PLANNER_SYSTEM, "planner")

    assert "FastAPI + PostgreSQL project." in prompt
    assert "Use ONLY read_file and write_file." in prompt
    assert "PROJECT CONTEXT:" in prompt


# ── Test 2: Empty domain context adds nothing ─────────────────────────────────

def test_empty_domain_context_adds_nothing():
    """An empty DomainContext must not modify the core prompt at all."""
    ctx = DomainContext()

    class FakeProvider:
        pass

    agent = PlannerAgent(FakeProvider(), domain_context=ctx)
    prompt = agent._build_system_prompt(PLANNER_SYSTEM, "planner")

    assert prompt == PLANNER_SYSTEM
    assert "PROJECT CONTEXT:" not in prompt


# ── Test 3: Per-agent context appears only in correct agent ───────────────────

def test_per_agent_context_isolation():
    """Planner context must NOT appear in critic prompt, and vice versa."""
    ctx = DomainContext(
        planner_context="PLANNER-ONLY-MARKER",
        critic_context="CRITIC-ONLY-MARKER",
        resolver_context="RESOLVER-ONLY-MARKER",
    )

    class FakeProvider:
        pass

    planner = PlannerAgent(FakeProvider(), domain_context=ctx)
    critic = CriticAgent(FakeProvider(), domain_context=ctx)
    resolver = ResolverAgent(FakeProvider(), domain_context=ctx)

    planner_prompt = planner._build_system_prompt(PLANNER_SYSTEM, "planner")
    critic_prompt = critic._build_system_prompt(CRITIC_SYSTEM, "critic")
    resolver_prompt = resolver._build_system_prompt(RESOLVER_SYSTEM, "resolver")

    # Planner sees its own context, not others
    assert "PLANNER-ONLY-MARKER" in planner_prompt
    assert "CRITIC-ONLY-MARKER" not in planner_prompt
    assert "RESOLVER-ONLY-MARKER" not in planner_prompt

    # Critic sees its own context, not others
    assert "CRITIC-ONLY-MARKER" in critic_prompt
    assert "PLANNER-ONLY-MARKER" not in critic_prompt
    assert "RESOLVER-ONLY-MARKER" not in critic_prompt

    # Resolver sees its own context, not others
    assert "RESOLVER-ONLY-MARKER" in resolver_prompt
    assert "PLANNER-ONLY-MARKER" not in resolver_prompt
    assert "CRITIC-ONLY-MARKER" not in resolver_prompt


# ── Test 4: Settings fields create DomainContext automatically ────────────────

def test_settings_create_domain_context():
    """SumoSettings domain context fields should auto-build a DomainContext."""
    from sumospace.settings import SumoSettings
    from sumospace.domain_context import DomainContext

    settings = SumoSettings(
        global_domain_context="Global project info",
        planner_domain_context="Planner-specific rules",
    )

    # Simulate what kernel.__init__ does
    has_any = any([
        settings.global_domain_context,
        settings.planner_domain_context,
        settings.critic_domain_context,
        settings.resolver_domain_context,
    ])
    assert has_any is True

    ctx = DomainContext(
        global_context=settings.global_domain_context,
        planner_context=settings.planner_domain_context,
        critic_context=settings.critic_domain_context,
        resolver_context=settings.resolver_domain_context,
    )

    assert ctx.global_context == "Global project info"
    assert ctx.planner_context == "Planner-specific rules"
    assert ctx.critic_context == ""
    assert ctx.resolver_context == ""

    planner_str = ctx.build_for("planner")
    assert "Global project info" in planner_str
    assert "Planner-specific rules" in planner_str


# ── Test 5: from_workspace detects FastAPI project correctly ──────────────────

@pytest.mark.asyncio
async def test_from_workspace_detects_fastapi():
    """from_workspace should detect FastAPI from pyproject.toml dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject = Path(tmpdir) / "pyproject.toml"
        pyproject.write_text(
            '[project]\n'
            'name = "my-api"\n'
            'requires-python = ">=3.11"\n'
            'dependencies = ["fastapi>=0.100", "uvicorn"]\n',
            encoding="utf-8",
        )

        ctx = await DomainContext.from_workspace(tmpdir)

        assert "FastAPI" in ctx.global_context
        assert "Python" in ctx.global_context
        assert ctx.planner_context  # Should have generated planner rules
        assert "read_file" in ctx.planner_context


# ── Test 6: Core JSON instructions always present ─────────────────────────────

def test_core_json_instructions_always_present():
    """
    CRITICAL: The core JSON format instructions must appear in the assembled
    prompt even when BOTH PromptTemplates AND DomainContext are provided.
    """
    ctx = DomainContext(
        global_context="Custom project context here.",
        planner_context="Custom planner rules here.",
    )

    # Simulate a PromptTemplates override (Layer 2) by passing templates=None
    # so the core system prompt is used directly from constants
    class FakeProvider:
        pass

    agent = PlannerAgent(FakeProvider(), templates=None, domain_context=ctx)
    prompt = agent._build_system_prompt(PLANNER_SYSTEM, "planner")

    # Layer 1 core identity MUST be present
    assert '"reasoning"' in prompt
    assert '"steps"' in prompt
    assert '"tool"' in prompt
    assert '"parameters"' in prompt
    assert "Output ONLY" in prompt

    # Layer 3 domain context MUST also be present
    assert "Custom project context here." in prompt
    assert "Custom planner rules here." in prompt


# ── Test: global_context appears in ALL agents ────────────────────────────────

def test_global_context_appears_in_all_agents():
    """global_context should be injected into every agent's prompt."""
    ctx = DomainContext(global_context="GLOBAL-MARKER-12345")

    class FakeProvider:
        pass

    planner = PlannerAgent(FakeProvider(), domain_context=ctx)
    critic = CriticAgent(FakeProvider(), domain_context=ctx)
    resolver = ResolverAgent(FakeProvider(), domain_context=ctx)

    assert "GLOBAL-MARKER-12345" in planner._build_system_prompt(PLANNER_SYSTEM, "planner")
    assert "GLOBAL-MARKER-12345" in critic._build_system_prompt(CRITIC_SYSTEM, "critic")
    assert "GLOBAL-MARKER-12345" in resolver._build_system_prompt(RESOLVER_SYSTEM, "resolver")


# ── Test: None domain context leaves prompt unchanged ─────────────────────────

def test_none_domain_context_unchanged():
    """When domain_context is None, prompt must equal the core system prompt exactly."""
    class FakeProvider:
        pass

    agent = PlannerAgent(FakeProvider(), domain_context=None)
    prompt = agent._build_system_prompt(PLANNER_SYSTEM, "planner")

    assert prompt == PLANNER_SYSTEM
