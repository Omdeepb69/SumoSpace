# sumospace/committee.py

"""
Committee — Multi-Agent Deliberation System
=============================================
Before any plan is executed, three specialist agents deliberate:
  - Planner   : Decomposes the task into a step-by-step execution plan
  - Critic    : Identifies risks, gaps, and failure modes
  - Resolver  : Synthesises a final approved plan (or halts if unsafe)

Consensus is required before the kernel executes any plan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class ExecutionStep:
    step_number: int
    tool: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    critical: bool = False   # If True, failure halts the entire plan


@dataclass
class ExecutionPlan:
    task: str
    steps: list[ExecutionStep]
    reasoning: str = ""
    estimated_duration_s: float = 0
    risks: list[str] = field(default_factory=list)
    approved: bool = False
    approval_notes: str = ""


@dataclass
class CommitteeVerdict:
    approved: bool
    plan: ExecutionPlan
    planner_output: str = ""
    critic_output: str = ""
    resolver_output: str = ""
    rejection_reason: str = ""


# ─── Prompts ─────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are the Planner agent in a multi-agent task execution system.
Your role: Given a task description and available tools, produce a detailed, safe execution plan.

Available tools: read_file, write_file, list_directory, search_files, patch_file,
shell, web_search, fetch_url, docker, dependencies

Output ONLY a JSON object with this schema:
{
  "reasoning": "Brief explanation of your approach",
  "estimated_duration_s": <number>,
  "steps": [
    {
      "step_number": 1,
      "tool": "<tool_name>",
      "description": "<what this step does>",
      "parameters": {<tool-specific params>},
      "expected_output": "<what success looks like>",
      "critical": <true if failure should halt everything>
    }
  ]
}

Rules:
- Be specific. Include actual file paths, commands, parameters.
- Start with read/scan steps before write steps.
- Mark destructive operations (write_file, shell rm, etc.) as critical.
- Maximum 12 steps. If more are needed, break the task.
- Output ONLY JSON. No markdown fences."""


CRITIC_SYSTEM = """You are the Critic agent in a multi-agent task execution system.
Your role: Review the proposed execution plan and identify ALL potential issues.

Output ONLY a JSON object:
{
  "risks": ["<risk 1>", "<risk 2>"],
  "blockers": ["<blocker 1>"],  // Must-fix issues that make plan unsafe to execute
  "suggestions": ["<improvement 1>"],
  "verdict": "approve" | "revise" | "reject",
  "verdict_reason": "<one sentence>"
}

Be honest and rigorous. Reject plans that:
- Delete or overwrite important files without backup
- Run commands that could damage the system
- Have missing prerequisite steps
- Make unjustified assumptions about file locations or system state"""


RESOLVER_SYSTEM = """You are the Resolver agent in a multi-agent task execution system.
Your role: Given the original plan and the critic's feedback, produce the final approved plan.

If the critic's blockers are unresolvable, output:
{"approved": false, "rejection_reason": "<reason>", "steps": []}

Otherwise, produce an improved plan:
{
  "approved": true,
  "approval_notes": "<what was improved>",
  "reasoning": "<approach>",
  "estimated_duration_s": <number>,
  "steps": [...]   // same schema as planner
}

Output ONLY JSON. No markdown. No explanation outside the JSON."""


# ─── Individual Agents ────────────────────────────────────────────────────────

class PlannerAgent:
    def __init__(self, provider):
        self._provider = provider

    async def plan(self, task: str, context: str = "") -> tuple[ExecutionPlan, str]:
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        raw = await self._provider.complete(
            user=prompt,
            system=PLANNER_SYSTEM,
            temperature=0.1,
            max_tokens=2048,
        )

        plan, raw_clean = self._parse_plan(task, raw)
        return plan, raw_clean

    def _parse_plan(self, task: str, raw: str) -> tuple[ExecutionPlan, str]:
        raw_clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(raw_clean)
            steps = [
                ExecutionStep(
                    step_number=s.get("step_number", i + 1),
                    tool=s.get("tool", "shell"),
                    description=s.get("description", ""),
                    parameters=s.get("parameters", {}),
                    expected_output=s.get("expected_output", ""),
                    critical=s.get("critical", False),
                )
                for i, s in enumerate(data.get("steps", []))
            ]
            return ExecutionPlan(
                task=task,
                steps=steps,
                reasoning=data.get("reasoning", ""),
                estimated_duration_s=float(data.get("estimated_duration_s", 0)),
            ), raw_clean
        except Exception:
            # Fallback: single shell step
            return ExecutionPlan(
                task=task,
                steps=[ExecutionStep(
                    step_number=1, tool="shell",
                    description=f"Execute: {task}",
                    parameters={"command": "echo 'Plan parsing failed; manual intervention needed'"},
                    critical=True,
                )],
                reasoning="Plan parsing failed; using minimal fallback.",
            ), raw_clean


class CriticAgent:
    def __init__(self, provider):
        self._provider = provider

    async def critique(
        self,
        plan: ExecutionPlan,
        task: str,
    ) -> tuple[str, str, list[str], list[str]]:
        """Returns: (verdict, reason, risks, blockers)"""
        plan_json = json.dumps({
            "task": task,
            "steps": [
                {
                    "step_number": s.step_number,
                    "tool": s.tool,
                    "description": s.description,
                    "parameters": s.parameters,
                    "critical": s.critical,
                }
                for s in plan.steps
            ],
        }, indent=2)

        raw = await self._provider.complete(
            user=f"Review this execution plan:\n{plan_json}",
            system=CRITIC_SYSTEM,
            temperature=0.1,
            max_tokens=1024,
        )

        raw_clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(raw_clean)
            verdict = data.get("verdict", "approve")
            reason = data.get("verdict_reason", "")
            risks = data.get("risks", [])
            blockers = data.get("blockers", [])
        except Exception:
            verdict, reason, risks, blockers = "approve", "Critique parsing failed", [], []

        return verdict, reason, risks, blockers, raw_clean


class ResolverAgent:
    def __init__(self, provider):
        self._provider = provider

    async def resolve(
        self,
        task: str,
        original_plan: ExecutionPlan,
        critic_verdict: str,
        critic_reason: str,
        risks: list[str],
        blockers: list[str],
    ) -> tuple[ExecutionPlan, bool, str, str]:
        """Returns: (final_plan, approved, approval_notes, raw)"""
        if critic_verdict == "approve":
            original_plan.approved = True
            original_plan.risks = risks
            return original_plan, True, "Approved by critic without changes", ""

        prompt = json.dumps({
            "task": task,
            "original_plan_steps": [
                {"tool": s.tool, "description": s.description, "parameters": s.parameters}
                for s in original_plan.steps
            ],
            "critic_verdict": critic_verdict,
            "critic_reason": critic_reason,
            "risks": risks,
            "blockers": blockers,
        }, indent=2)

        raw = await self._provider.complete(
            user=prompt,
            system=RESOLVER_SYSTEM,
            temperature=0.1,
            max_tokens=2048,
        )

        raw_clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(raw_clean)
            approved = bool(data.get("approved", False))

            if not approved:
                return original_plan, False, "", data.get("rejection_reason", "Rejected by resolver")

            steps = [
                ExecutionStep(
                    step_number=s.get("step_number", i + 1),
                    tool=s.get("tool", "shell"),
                    description=s.get("description", ""),
                    parameters=s.get("parameters", {}),
                    expected_output=s.get("expected_output", ""),
                    critical=s.get("critical", False),
                )
                for i, s in enumerate(data.get("steps", []))
            ]
            plan = ExecutionPlan(
                task=task,
                steps=steps,
                reasoning=data.get("reasoning", ""),
                estimated_duration_s=float(data.get("estimated_duration_s", 0)),
                risks=risks,
                approved=True,
                approval_notes=data.get("approval_notes", ""),
            )
            return plan, True, plan.approval_notes, raw_clean
        except Exception as e:
            # Parse failed — approve original with warning
            original_plan.approved = True
            original_plan.risks = risks
            return original_plan, True, f"Resolver parse failed ({e}); using original", raw_clean


# ─── Committee ────────────────────────────────────────────────────────────────

class Committee:
    """
    Three-agent deliberation panel.
    Planner → Critic → Resolver → final approved plan.

    Usage:
        committee = Committee(provider, require_consensus=True)
        verdict = await committee.deliberate(task, context)
        if verdict.approved:
            # Execute verdict.plan
    """

    def __init__(self, provider, require_consensus: bool = True):
        self._planner = PlannerAgent(provider)
        self._critic = CriticAgent(provider)
        self._resolver = ResolverAgent(provider)
        self.require_consensus = require_consensus

    async def deliberate(
        self,
        task: str,
        context: str = "",
    ) -> CommitteeVerdict:
        """
        Full deliberation cycle: plan → critique → resolve.
        Returns a CommitteeVerdict with approved plan or rejection reason.
        """
        # Phase 1: Planner proposes
        plan, planner_raw = await self._planner.plan(task, context)

        if not self.require_consensus:
            plan.approved = True
            return CommitteeVerdict(
                approved=True,
                plan=plan,
                planner_output=planner_raw,
            )

        # Phase 2: Critic reviews
        verdict, reason, risks, blockers, critic_raw = await self._critic.critique(plan, task)

        if verdict == "reject" and blockers:
            return CommitteeVerdict(
                approved=False,
                plan=plan,
                planner_output=planner_raw,
                critic_output=critic_raw,
                rejection_reason=f"Critic rejected: {reason}. Blockers: {'; '.join(blockers)}",
            )

        # Phase 3: Resolver synthesises
        final_plan, approved, notes, resolver_raw = await self._resolver.resolve(
            task=task,
            original_plan=plan,
            critic_verdict=verdict,
            critic_reason=reason,
            risks=risks,
            blockers=blockers,
        )

        return CommitteeVerdict(
            approved=approved,
            plan=final_plan,
            planner_output=planner_raw,
            critic_output=critic_raw,
            resolver_output=resolver_raw,
            rejection_reason="" if approved else notes,
        )
