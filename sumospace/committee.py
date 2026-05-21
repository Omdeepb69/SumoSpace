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

from .parsing import parse_llm_json
from .schemas import ExecutionPlan, ExecutionStep, CritiqueVerdict, ResolverOutput, dereference_schema


# ─── Data Models ─────────────────────────────────────────────────────────────

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
Your role: Given a task description and the list of available tools, produce a detailed, safe execution plan.

CRITICAL RULES:
- You MUST use ONLY the exact tool names listed in the AVAILABLE TOOLS section below.
- Do NOT invent or hallucinate tool names. If no matching tool exists, do NOT include that step.
- Use RELATIVE file paths (e.g., 'utils.py', not '/path/to/utils.py'). The workspace IS the current directory.
- Be specific. Include actual file paths, commands, parameters.
- Start with read/scan steps before write steps.
- To inspect code or find functions, ALWAYS use 'read_file' to examine file contents. Do NOT use 'search_files' with regexes to find code/functions, as LLM regexes are often incorrect.
- Mark destructive operations (write_file, shell rm, etc.) as critical.
- Maximum 12 steps. If more are needed, break the task.
- For file edits, prefer 'replace_text' (surgical edit) over 'write_file' (full overwrite).
- The 'replace_text' tool requires: path, old_text (exact text to find), new_text (replacement).
- The 'write_file' tool requires: path, content (full file content)."""


CRITIC_SYSTEM = """You are the Critic agent in a multi-agent task execution system.
Your role: Review the proposed execution plan and identify ALL potential issues.

Be constructive. Your goal is to IMPROVE plans, not block them:
- Use "reject" ONLY for plans that pose genuine safety risks (data loss, system damage, destructive commands).
- For all other issues (wrong tool names, missing steps, path errors, suboptimal ordering), use "revise" and explain what needs to change.
- If the plan is reasonable and uses valid tools, use "approve".
- Do NOT reject a plan just because it could be slightly better.

Reject plans ONLY if they:
- Delete or overwrite important files without backup
- Run commands that could damage the system
- Are fundamentally unsafe or destructive"""


RESOLVER_SYSTEM = """You are the Resolver agent in a multi-agent task execution system.
Your role: Given the original plan and the critic's feedback, decide whether to approve or reject.

DECISION RULES:
1. If the critic said "revise" with suggestions but NO blockers → set approved to TRUE.
   Apply the suggestions by providing revised_steps, or approve the original plan as-is.
2. If the critic said "revise" WITH blockers → try to fix them. If fixable, set approved to TRUE with revised_steps.
3. If the critic said "reject" → set approved to FALSE only if the rejection reason is genuinely unresolvable.
4. DEFAULT TO APPROVED. Most plans are good enough to execute. Only reject if the plan is truly dangerous.

When approved, set has_revision to true and provide revised_steps if you changed anything.
When approved without changes, set has_revision to false and leave revised_steps empty."""


# ─── Individual Agents ────────────────────────────────────────────────────────

class BaseAgent:
    """Override this to add custom committee agents."""
    role: str = "base"

    def __init__(self, provider, templates=None, domain_context=None):
        self._provider = provider
        self._templates = templates
        self._domain_context = domain_context

    def _build_system_prompt(self, core_system: str, agent_role: str) -> str:
        """Assemble the 3-layer system prompt."""
        template_key = f"{agent_role}_prompt"
        system = (
            self._templates.raw(template_key) if self._templates
            else core_system
        )

        if self._domain_context is not None:
            domain_str = self._domain_context.build_for(agent_role)
            if domain_str:
                system += domain_str

        return system

    async def run(self, task: str, context: str, **kwargs) -> dict:
        raise NotImplementedError


class PlannerAgent(BaseAgent):
    role = "planner"

    async def plan(self, task: str, context: str = "") -> tuple[ExecutionPlan, str]:
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context}"
        prompt += "\n\nIMPORTANT: Use RELATIVE file paths. The workspace is the current directory."

        system = self._build_system_prompt(PLANNER_SYSTEM, "planner")

        # Extract available tools from context and inject into system prompt
        available_tool_names = []
        if context and "AVAILABLE TOOLS" in context:
            tools_section = context.split("=== AVAILABLE TOOLS ===")
            if len(tools_section) > 1:
                tools_text = tools_section[1].split("===")[0].strip()
                system += f"\n\nAVAILABLE TOOLS (use ONLY these exact names):\n{tools_text}"
                # Parse tool names for post-validation
                for line in tools_text.split("\n"):
                    line = line.strip()
                    if line.startswith("- ") and ":" in line:
                        tool_name = line.split(":")[0].strip("- ").strip()
                        available_tool_names.append(tool_name)

        schema = dereference_schema(ExecutionPlan.model_json_schema())

        last_raw = ""
        for attempt in range(3):
            try:
                raw = await self._provider.complete_structured(
                    user=prompt,
                    system=system,
                    schema=schema,
                    temperature=min(0.1 + (attempt * 0.1), 0.4),
                    max_tokens=2048,
                )
                last_raw = raw

                # Guard: empty string means grammar engine failed silently
                if not raw or not raw.strip():
                    raise ValueError(
                        f"Structured output returned empty string (attempt {attempt+1}/3). "
                        "Likely a grammar/schema incompatibility in the provider."
                    )

                plan = ExecutionPlan.model_validate_json(raw)
                plan.raw_output = raw
                plan.task = task

                # Post-generation: validate and fix tool names
                if available_tool_names:
                    plan = self._validate_tool_names(plan, available_tool_names)

                return plan, raw
            except Exception as e:
                # LEGACY FALLBACK
                import os
                if os.environ.get("DEBUG_PLANNER"):
                    with open("/tmp/planner_debug.log", "a") as f:
                        f.write(f"\n--- FALLBACK (Attempt {attempt+1}) ---\nError: {e}\nRaw:\n{last_raw}\n")
                
                # If structured parsing totally fails, attempt legacy repair logic
                try:
                    plan, _ = self._legacy_parse_plan(task, last_raw)
                    if plan.steps:
                        return plan, last_raw
                except Exception:
                    pass

        # Return a failed empty plan if all attempts failed
        return ExecutionPlan(
            protocol_version="1.0",
            task=task,
            reasoning="Plan parsing failed; halting to prevent unsafe fallback.",
            steps=[],
            raw_output=last_raw
        ), last_raw
    def _validate_tool_names(self, plan: ExecutionPlan, available: list[str]) -> ExecutionPlan:
        """Post-generation validation: fix hallucinated tool names via fuzzy matching."""
        import difflib

        # Common hallucination → real tool mapping
        TOOL_ALIASES = {
            "update_file": "write_file",
            "edit_file": "replace_text",
            "modify_file": "replace_text",
            "create_file": "write_file",
            "find_functions": "search_files",
            "grep": "search_files",
            "search": "search_files",
            "find": "search_files",
            "ls": "list_directory",
            "dir": "list_directory",
            "cat": "read_file",
            "run": "shell",
            "execute": "shell",
            "bash": "shell",
            "run_command": "shell",
            "exec": "shell",
        }

        valid_steps = []
        for step in plan.steps:
            tool = step.tool.strip()

            # Already valid
            if tool in available:
                valid_steps.append(step)
                continue

            # Check alias table
            if tool in TOOL_ALIASES and TOOL_ALIASES[tool] in available:
                step.tool = TOOL_ALIASES[tool]
                valid_steps.append(step)
                continue

            # Fuzzy match
            close = difflib.get_close_matches(tool, available, n=1, cutoff=0.6)
            if close:
                step.tool = close[0]
                valid_steps.append(step)
                continue

            # Unresolvable — substitute with the invalid tool placeholder
            # This preserves chronological step numbering and provides explicit failure feedback
            step.parameters = {"hallucinated_tool": tool}
            step.tool = "invalid_tool"
            valid_steps.append(step)

        plan.steps = valid_steps
        return plan

    def _legacy_parse_plan(self, task: str, raw: str) -> tuple[ExecutionPlan, str]:
        # LEGACY FALLBACK
        try:
            data, repair_used = parse_llm_json(raw, expected_keys=["steps"])
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
                protocol_version="1.0",
                task=task,
                steps=steps,
                reasoning=data.get("reasoning", ""),
                estimated_duration_s=int(data.get("estimated_duration_s", 30)),
                raw_output=raw,
            ), raw
        except Exception:
            raise


class CriticAgent(BaseAgent):
    role = "critic"

    async def critique(
        self,
        plan: ExecutionPlan,
        task: str,
    ) -> tuple[str, str, list[str], list[str], str]:
        """Returns: (verdict, reason, risks, blockers, raw)"""
        plan_json = plan.model_dump_json(exclude={"raw_output", "approved", "approval_notes", "risks"})

        system = self._build_system_prompt(CRITIC_SYSTEM, "critic")
        schema = dereference_schema(CritiqueVerdict.model_json_schema())

        last_raw = ""
        for attempt in range(3):
            try:
                raw = await self._provider.complete_structured(
                    user=f"Review this execution plan:\n{plan_json}",
                    system=system,
                    schema=schema,
                    temperature=min(0.1 + (attempt * 0.1), 0.4),
                    max_tokens=1024,
                )
                last_raw = raw

                # Guard: empty string means grammar engine failed silently
                if not raw or not raw.strip():
                    raise ValueError(
                        f"Structured output returned empty string (attempt {attempt+1}/3). "
                        "Likely a grammar/schema incompatibility in the provider."
                    )

                verdict_model = CritiqueVerdict.model_validate_json(raw)
                return (
                    verdict_model.verdict, 
                    verdict_model.reason, 
                    verdict_model.risks, 
                    verdict_model.blockers, 
                    raw
                )
            except Exception as e:
                # LEGACY FALLBACK
                try:
                    data, _ = parse_llm_json(last_raw, expected_keys=["verdict"])
                    return (
                        data.get("verdict", "approve"),
                        data.get("verdict_reason", ""),
                        data.get("risks", []),
                        data.get("blockers", []),
                        last_raw
                    )
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
        # Fast-path: critic approved
        if critic_verdict == "approve":
            original_plan.approved = True
            original_plan.risks = risks
            return original_plan, True, "Approved by critic without changes", ""

        # Fast-path: critic said "revise" with zero blockers
        # The plan is safe — suggestions are optional improvements, not dealbreakers
        if critic_verdict == "revise" and not blockers:
            original_plan.approved = True
            original_plan.risks = risks
            original_plan.approval_notes = f"Auto-approved (revise with no blockers). Critic note: {critic_reason}"
            return original_plan, True, original_plan.approval_notes, ""

        prompt = json.dumps({
            "task": task,
            "original_plan_steps": original_plan.model_dump(include={"steps"})["steps"],
            "critic_verdict": critic_verdict,
            "critic_reason": critic_reason,
            "risks": risks,
            "blockers": blockers,
        }, indent=2)

        base_system = self._build_system_prompt(RESOLVER_SYSTEM, "resolver")
        schema = dereference_schema(ResolverOutput.model_json_schema())
        
        last_raw = ""
        for attempt in range(3):
            try:
                raw = await self._provider.complete_structured(
                    user=prompt,
                    system=base_system,
                    schema=schema,
                    temperature=min(0.1 + (attempt * 0.1), 0.4),
                    max_tokens=2048,
                )
                last_raw = raw

                # Guard: empty string means grammar engine failed silently
                if not raw or not raw.strip():
                    raise ValueError(
                        f"Structured output returned empty string (attempt {attempt+1}/3). "
                        "Likely a grammar/schema incompatibility in the provider."
                    )

                resolver_model = ResolverOutput.model_validate_json(raw)
                
                if not resolver_model.approved:
                    return original_plan, False, "", resolver_model.rejection_reason

                # Reconstruct plan from revised_steps if resolver made revisions
                if resolver_model.has_revision and resolver_model.revised_steps:
                    final_plan = ExecutionPlan(
                        protocol_version="1.0",
                        task=task,
                        steps=resolver_model.revised_steps,
                        reasoning=original_plan.reasoning,
                        estimated_duration_s=original_plan.estimated_duration_s,
                        risks=risks,
                        approved=True,
                        approval_notes=resolver_model.approval_notes,
                        raw_output=raw,
                    )
                else:
                    final_plan = original_plan
                    final_plan.approved = True
                    final_plan.approval_notes = resolver_model.approval_notes
                    final_plan.risks = risks
                    final_plan.raw_output = raw
                
                return final_plan, True, final_plan.approval_notes, raw
                
            except Exception as e:
                # LEGACY FALLBACK
                if not last_raw or not last_raw.strip():
                    if attempt < 2:
                        continue
                    return original_plan, False, "", (
                        f"Resolver returned empty output on all attempts. "
                        "Provider grammar engine likely cannot handle the schema."
                    )

                try:
                    data, _ = parse_llm_json(last_raw, expected_keys=["approved"])
                    approved = bool(data.get("approved", False))

                    if not approved:
                        rejection = data.get("rejection_reason", "Rejected by resolver")
                        return original_plan, False, "", rejection

                    steps = [
                        ExecutionStep(
                            step_number=s.get("step_number", i + 1),
                            tool=s.get("tool", "shell"),
                            description=s.get("description", ""),
                            parameters=s.get("parameters", {}),
                            expected_output=s.get("expected_output", ""),
                            critical=s.get("critical", False),
                        )
                        for i, s in enumerate(data.get("steps", data.get("revised_steps", [])))
                    ]
                    plan = ExecutionPlan(
                        protocol_version="1.0",
                        task=task,
                        steps=steps,
                        reasoning=data.get("reasoning", ""),
                        estimated_duration_s=int(data.get("estimated_duration_s", 30)),
                        risks=risks,
                        approved=True,
                        approval_notes=data.get("approval_notes", ""),
                        raw_output=last_raw,
                    )
                    return plan, True, plan.approval_notes, last_raw
                except Exception:
                    if attempt < 2:
                        continue
                    return original_plan, False, "", (
                        f"Resolver output unparseable ({e}). "
                        "Refusing to approve a critic-flagged plan with unverifiable resolution."
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
        domain_context=None,
        custom_agents: list[BaseAgent] | None = None,
        planner: PlannerAgent | None = None,
        critic: CriticAgent | None = None,
        resolver: ResolverAgent | None = None,
    ):
        provider_to_use = planning_provider or provider
        self._planner = planner or PlannerAgent(provider_to_use, templates=templates, domain_context=domain_context)
        self._critic = critic or CriticAgent(provider_to_use, templates=templates, domain_context=domain_context)
        self._resolver = resolver or ResolverAgent(provider_to_use, templates=templates, domain_context=domain_context)
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
