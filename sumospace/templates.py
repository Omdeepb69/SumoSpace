# sumospace/templates.py

"""
PromptTemplate system for SumoSpace.
=====================================
Allows users to override any built-in prompt by placing .txt files in a
template directory specified via ``SumoSettings.prompt_template_path``.

Rendering uses Python ``str.format_map()`` with a ``SafeFormatMap`` that
returns empty strings for missing keys — no Jinja2, no dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# ── Variable schema per template ──────────────────────────────────────────────
# Documents exactly which variables each template may reference.

TEMPLATE_VARIABLES: dict[str, list[str]] = {
    "system_prompt":    ["version"],
    "planner_prompt":   [],
    "critic_prompt":    [],
    "resolver_prompt":  [],
    "synthesis_prompt": ["task", "step_outputs", "context"],
}

# ── Default built-in prompts ─────────────────────────────────────────────────

_DEFAULTS: dict[str, str] = {
    "system_prompt": (
        "You are Sumo, an advanced AI assistant built on SumoSpace {version}. "
        "You help users by planning and executing multi-step tasks using "
        "available tools. Be precise, factual, and concise."
    ),
    "planner_prompt": (
        "You are the Planner agent in a multi-agent task execution system.\n"
        "Your role: Given a task description and the list of available tools "
        "in the context, produce a detailed, safe execution plan.\n\n"
        "Output ONLY a JSON object with this schema:\n"
        "{\n"
        '  "reasoning": "Brief explanation of your approach",\n'
        '  "estimated_duration_s": <number>,\n'
        '  "steps": [\n'
        "    {\n"
        '      "step_number": 1,\n'
        '      "tool": "<tool_name>",\n'
        '      "description": "<what this step does>",\n'
        '      "parameters": {<tool-specific params>},\n'
        '      "expected_output": "<what success looks like>",\n'
        '      "critical": <true if failure should halt everything>\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use ONLY the tools listed in the \"AVAILABLE TOOLS\" section of the context.\n"
        "- Be specific. Include actual file paths, commands, parameters.\n"
        "- Start with read/scan steps before write steps.\n"
        "- Mark destructive operations (write_file, shell rm, etc.) as critical.\n"
        "- Maximum 12 steps. If more are needed, break the task.\n"
        "- Output ONLY JSON. No markdown fences."
    ),
    "critic_prompt": (
        "You are the Critic agent in a multi-agent task execution system.\n"
        "Your role: Review the proposed execution plan and identify ALL potential issues.\n\n"
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "risks": ["<risk 1>", "<risk 2>"],\n'
        '  "blockers": ["<blocker 1>"],  // Must-fix issues that make plan unsafe to execute\n'
        '  "suggestions": ["<improvement 1>"],\n'
        '  "verdict": "approve" | "revise" | "reject",\n'
        '  "verdict_reason": "<one sentence>"\n'
        "}\n\n"
        "Be honest and rigorous. Reject plans that:\n"
        "- Delete or overwrite important files without backup\n"
        "- Run commands that could damage the system\n"
        "- Have missing prerequisite steps\n"
        "- Make unjustified assumptions about file locations or system state"
    ),
    "resolver_prompt": (
        "You are the Resolver agent in a multi-agent task execution system.\n"
        "Your role: Given the original plan and the critic's feedback, produce the final approved plan.\n\n"
        "If the critic's blockers are unresolvable, output:\n"
        '{"approved": false, "rejection_reason": "<reason>", "steps": []}\n\n'
        "Otherwise, produce an improved plan:\n"
        "{\n"
        '  "approved": true,\n'
        '  "approval_notes": "<what was improved>",\n'
        '  "reasoning": "<approach>",\n'
        '  "estimated_duration_s": <number>,\n'
        '  "steps": [...]   // same schema as planner\n'
        "}\n\n"
        "Output ONLY JSON. No markdown. No explanation outside the JSON."
    ),
    "synthesis_prompt": (
        "You are a helpful assistant summarising completed tasks.\n"
        "Given the task and tool outputs, write a clear, concise summary of what was done.\n"
        "Be specific. Mention files changed, commands run, or answers found.\n\n"
        "Task: {task}\n\n"
        "Tool outputs:\n{step_outputs}\n\n"
        "Context used:\n{context}"
    ),
}

# ── Format helpers ────────────────────────────────────────────────────────────


class SafeFormatMap(dict):
    """Dict subclass that returns '' for missing keys instead of raising KeyError.

    This allows templates to reference optional variables without crashing::

        >>> "Hello {name}, your {role}".format_map(SafeFormatMap({"name": "Alice"}))
        "Hello Alice, your "
    """

    def __missing__(self, key: str) -> str:
        return ""


# ── TemplateManager ───────────────────────────────────────────────────────────


class TemplateManager:
    """
    Loads and renders prompt templates.

    Priority:
        1. Custom templates from ``template_path`` directory
        2. Built-in defaults in ``_DEFAULTS``

    Custom templates are ``.txt`` files named after the template key,
    e.g. ``planner_prompt.txt``.
    """

    def __init__(self, template_path: str | None = None):
        self._templates: dict[str, str] = dict(_DEFAULTS)
        if template_path:
            self._load_from_path(template_path)

    def _load_from_path(self, path: str) -> None:
        """Load custom templates from a directory, validating variables."""
        p = Path(path)
        if not p.is_dir():
            console.print(f"[yellow]Template path '{path}' is not a directory, skipping.[/yellow]")
            return

        for template_file in p.glob("*.txt"):
            name = template_file.stem
            content = template_file.read_text(encoding="utf-8")

            if name not in TEMPLATE_VARIABLES:
                console.print(
                    f"[yellow]Unknown template '{name}' in {path}. "
                    f"Valid templates: {sorted(TEMPLATE_VARIABLES.keys())}[/yellow]"
                )
                continue

            # Validate referenced variables at load time
            unknown = self._find_unknown_vars(content, TEMPLATE_VARIABLES[name])
            if unknown:
                console.print(
                    f"[yellow]Template '{name}' references unknown variables: {unknown}. "
                    f"Available: {TEMPLATE_VARIABLES[name]}[/yellow]"
                )

            self._templates[name] = content
            console.print(f"[dim]Loaded custom template: {name}[/dim]")

    @staticmethod
    def _find_unknown_vars(template: str, known: list[str]) -> list[str]:
        """Find format variables in template that aren't in the known list."""
        # Match {word} but not {{escaped}} — simple regex for .format() placeholders
        referenced = set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", template))
        return sorted(referenced - set(known))

    def get(self, name: str, **variables: Any) -> str:
        """
        Render a template by name with the given variables.

        Missing variables are silently replaced with empty strings.
        Unknown template names fall through to an empty string with a warning.
        """
        template = self._templates.get(name)
        if template is None:
            console.print(f"[yellow]No template found for '{name}'[/yellow]")
            return ""
        return template.format_map(SafeFormatMap(variables))

    def raw(self, name: str) -> str | None:
        """Return the raw template string without rendering, or None."""
        return self._templates.get(name)

    @property
    def available(self) -> list[str]:
        """List all loaded template names."""
        return sorted(self._templates.keys())
