# tests/test_committee.py

import pytest
from sumospace.committee import (
    PlannerAgent,
    CriticAgent,
    ResolverAgent,
    Committee,
    ExecutionStep,
    ExecutionPlan,
    CommitteeVerdict,
)


def make_plan(task="test task", steps=None):
    steps = steps or [
        ExecutionStep(
            step_number=1, tool="shell",
            description="Run echo",
            parameters={"command": "echo hello"},
            critical=False,
        )
    ]
    return ExecutionPlan(task=task, steps=steps)


@pytest.mark.asyncio
class TestPlannerAgent:
    async def test_plan_returns_execution_plan(self, mock_provider):
        agent = PlannerAgent(mock_provider)
        plan, raw = await agent.plan("List all Python files")
        assert isinstance(plan, ExecutionPlan)
        assert plan.task == "List all Python files"
        assert len(plan.steps) >= 1

    async def test_plan_with_context(self, mock_provider):
        agent = PlannerAgent(mock_provider)
        plan, raw = await agent.plan("Fix the bug", context="Error: NoneType in line 42")
        assert isinstance(plan, ExecutionPlan)

    async def test_plan_step_has_required_fields(self, mock_provider):
        agent = PlannerAgent(mock_provider)
        plan, _ = await agent.plan("Read file src/main.py")
        step = plan.steps[0]
        assert hasattr(step, "tool")
        assert hasattr(step, "description")
        assert hasattr(step, "parameters")
        assert hasattr(step, "critical")

    async def test_plan_fallback_on_bad_json(self):
        """Provider that returns garbage JSON falls back gracefully."""
        class GarbageProvider:
            async def complete(self, **kwargs):
                return "not json at all ~~~ {{{}}"
            async def initialize(self): pass

        agent = PlannerAgent(GarbageProvider())
        plan, raw = await agent.plan("some task")
        assert len(plan.steps) == 0
        assert "parsing failed" in plan.reasoning.lower()


@pytest.mark.asyncio
class TestCriticAgent:
    async def test_critique_approve(self, mock_provider):
        agent = CriticAgent(mock_provider)
        plan = make_plan()
        verdict, reason, risks, blockers, raw = await agent.critique(plan, "test task")
        assert verdict in ("approve", "revise", "reject")

    async def test_critique_returns_lists(self, mock_provider):
        agent = CriticAgent(mock_provider)
        plan = make_plan()
        _, _, risks, blockers, _ = await agent.critique(plan, "test task")
        assert isinstance(risks, list)
        assert isinstance(blockers, list)

    async def test_critique_fallback_on_bad_json(self):
        class BadProvider:
            async def complete(self, **kwargs):
                return "totally invalid"
            async def initialize(self): pass

        agent = CriticAgent(BadProvider())
        plan = make_plan()
        verdict, reason, risks, blockers, _ = await agent.critique(plan, "task")
        # Should fall back gracefully
        assert verdict == "approve"  # fallback
        assert isinstance(risks, list)


@pytest.mark.asyncio
class TestResolverAgent:
    async def test_resolve_approve_when_critic_approves(self, mock_provider):
        agent = ResolverAgent(mock_provider)
        plan = make_plan()
        final, approved, notes, raw = await agent.resolve(
            task="test", original_plan=plan,
            critic_verdict="approve", critic_reason="looks good",
            risks=[], blockers=[],
        )
        assert approved is True
        assert final.approved is True

    async def test_resolve_calls_llm_when_critic_revises(self, mock_provider):
        agent = ResolverAgent(mock_provider)
        plan = make_plan()
        final, approved, notes, raw = await agent.resolve(
            task="test", original_plan=plan,
            critic_verdict="revise", critic_reason="needs improvement",
            risks=["risk 1"], blockers=[],
        )
        assert isinstance(approved, bool)

    async def test_resolve_reject_on_unresolvable(self):
        """Provider returns rejected JSON."""
        class RejectProvider:
            async def complete(self, **kwargs):
                return '{"approved": false, "rejection_reason": "too dangerous"}'
            async def initialize(self): pass

        agent = ResolverAgent(RejectProvider())
        plan = make_plan()
        _, approved, _, rejection = await agent.resolve(
            task="rm -rf everything", original_plan=plan,
            critic_verdict="reject", critic_reason="dangerous",
            risks=["data loss"], blockers=["destroys system"],
        )
        assert approved is False


@pytest.mark.asyncio
class TestCommittee:
    async def test_deliberate_approves_safe_task(self, mock_provider):
        committee = Committee(mock_provider, require_consensus=True)
        verdict = await committee.deliberate("List all Python files in src/")
        assert isinstance(verdict, CommitteeVerdict)
        assert verdict.approved is True
        assert len(verdict.plan.steps) >= 1

    async def test_deliberate_skips_consensus_when_disabled(self, mock_provider):
        committee = Committee(mock_provider, require_consensus=False)
        verdict = await committee.deliberate("Some task")
        assert verdict.approved is True
        assert verdict.critic_output == ""  # Critic not called

    async def test_deliberate_returns_plan(self, mock_provider):
        committee = Committee(mock_provider)
        verdict = await committee.deliberate("Read the README.md file")
        assert verdict.plan is not None
        assert verdict.plan.task == "Read the README.md file"

    async def test_deliberate_populates_outputs(self, mock_provider):
        committee = Committee(mock_provider, require_consensus=True)
        verdict = await committee.deliberate("Write tests for the auth module")
        assert verdict.planner_output  # Planner always runs
        # Critic and resolver run only with consensus

    async def test_rejection_when_critic_blocks(self):
        """Simulate a critic that always rejects with blockers."""
        import json

        call_count = [0]

        class StrictProvider:
            async def complete(self, user="", system="", **kwargs):
                call_count[0] += 1
                # First call: planner
                if "steps" in system and "planner" in system.lower():
                    return json.dumps({
                        "reasoning": "plan",
                        "estimated_duration_s": 1,
                        "steps": [{"step_number": 1, "tool": "shell",
                                   "description": "dangerous", "parameters": {"command": "rm -rf /"},
                                   "critical": True}]
                    })
                # Second call: critic — always reject with blockers
                if "risks" in system.lower() or "critic" in system.lower():
                    return json.dumps({
                        "risks": ["destroys filesystem"],
                        "blockers": ["rm -rf / is catastrophic"],
                        "suggestions": [],
                        "verdict": "reject",
                        "verdict_reason": "unacceptably dangerous",
                    })
                return "{}"
            async def initialize(self): pass

        committee = Committee(StrictProvider(), require_consensus=True)
        verdict = await committee.deliberate("Delete everything")
        assert verdict.approved is False
        assert "reject" in verdict.rejection_reason.lower() or "block" in verdict.rejection_reason.lower()
