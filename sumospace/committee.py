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

def _clean_json(raw: str) -> str:
    """Strip markdown fences and leading/trailing noise before parsing."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Find first { and last } in case of leading text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return raw


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
Your role: Given a task description and the list of available tools in the context, produce a detailed, safe execution plan.

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
- Use ONLY the tools listed in the "AVAILABLE TOOLS" section of the context.
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

class BaseAgent:
    """Override this to add custom committee agents."""
    role: str = "base"

    def __init__(self, provider, templates=None):
        self._provider = provider
        self._templates = templates

    async def run(self, task: str, context: str, **kwargs) -> dict:
        raise NotImplementedError

class PlannerAgent(BaseAgent):
    role = "planner"

    async def plan(self, task: str, context: str = "") -> tuple[ExecutionPlan, str]:
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        system = (
            self._templates.raw("planner_prompt") if self._templates
            else PLANNER_SYSTEM
        )

        last_raw = ""
        for attempt in range(3):
            raw = await self._provider.complete(
                user=prompt,
                system=system,
                temperature=min(0.1 + (attempt * 0.1), 0.4),
                max_tokens=2048,
            )
            last_raw = raw
            plan, raw_clean = self._parse_plan(task, raw)
            if plan.steps or attempt == 2:
                return plan, raw_clean
        
        return self._parse_plan(task, last_raw)

    def _parse_plan(self, task: str, raw: str) -> tuple[ExecutionPlan, str]:
        # Strip markdown fences and noise
        raw_clean = _clean_json(raw)
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
            return ExecutionPlan(
                task=task,
                steps=[],
                reasoning="Plan parsing failed; halting to prevent unsafe fallback.",
            ), raw_clean


class CriticAgent(BaseAgent):
    role = "critic"

    async def critique(
        self,
        plan: ExecutionPlan,
        task: str,
    ) -> tuple[str, str, list[str], list[str], str]:
        """Returns: (verdict, reason, risks, blockers, raw)"""
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

        system = (
            self._templates.raw("critic_prompt") if self._templates
            else CRITIC_SYSTEM
        )

        last_raw = ""
        for attempt in range(3):
            raw = await self._provider.complete(
                user=f"Review this execution plan:\n{plan_json}",
                system=system,
                temperature=min(0.1 + (attempt * 0.1), 0.4),
                max_tokens=1024,
            )
            last_raw = raw
            raw_clean = _clean_json(raw)
            try:
                data = json.loads(raw_clean)
                verdict = data.get("verdict", "approve")
                reason = data.get("verdict_reason", "")
                risks = data.get("risks", [])
                blockers = data.get("blockers", [])
                return verdict, reason, risks, blockers, raw_clean
            except Exception:
                if attempt == 2:
                    return "approve", "Critique parsing failed", [], [], last_raw
                continue


class ResolverAgent(BaseAgent):
    role = "resolver"

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

        raw_clean = ""
        base_system = (
            self._templates.raw("resolver_prompt") if self._templates
            else RESOLVER_SYSTEM
        )
        for attempt in range(3):
            if attempt > 0:
                system_prompt = base_system + f"\nCRITICAL: Your previous response was not valid JSON. (Attempt {attempt+1}/3)\nOutput ONLY a raw JSON object. No explanation. No markdown. No backticks.\nStart your response with {{ and end with }}."
            else:
                system_prompt = base_system

            raw = await self._provider.complete(
                user=prompt,
                system=system_prompt,
                temperature=min(0.1 + (attempt * 0.1), 0.4),
                max_tokens=2048,
            )

            raw_clean = _clean_json(raw)
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
                if attempt < 2:
                    continue
                # A parse failure on a flagged plan is NOT approval.
                return original_plan, False, "", (
                    f"Resolver output unparseable ({e}). "
                    "Refusing to approve a critic-flagged plan with unverifiable resolution. "
                    "Retry or use --no-consensus to bypass."
                )


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

    def __init__(
        self,
        provider,
        planning_provider=None,
        require_consensus: bool = True,
        templates=None,
        custom_agents: list[BaseAgent] | None = None,
        planner: PlannerAgent | None = None,
        critic: CriticAgent | None = None,
        resolver: ResolverAgent | None = None,
    ):
        provider_to_use = planning_provider or provider
        self._planner = planner or PlannerAgent(provider_to_use, templates=templates)
        self._critic = critic or CriticAgent(provider_to_use, templates=templates)
        self._resolver = resolver or ResolverAgent(provider_to_use, templates=templates)
        self._custom_agents = custom_agents or []
        self.require_consensus = require_consensus

    async def deliberate(
        self,
        task: str,
        context: str = "",
        mode: str = "full",
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

        if mode == "plan_only":
            return CommitteeVerdict(
                approved=True,
                plan=plan,
                planner_output=planner_raw,
                rejection_reason="plan_only mode — critique skipped",
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
            
        if mode == "critique_only":
            return CommitteeVerdict(
                approved=True,
                plan=plan,
                planner_output=planner_raw,
                critic_output=critic_raw,
                rejection_reason="critique_only mode — resolver skipped",
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
