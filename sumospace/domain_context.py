# sumospace/domain_context.py

"""
DomainContext — Additive Prompt Layering (Layer 3)
===================================================
Injects project-specific domain knowledge into agent prompts
WITHOUT replacing core JSON format instructions (Layer 1) or
persona templates (Layer 2).

All fields are optional. Empty strings are never injected.

Usage::

    context = DomainContext(
        global_context="FastAPI + PostgreSQL project. Python 3.11. All async.",
        planner_context="For file edits: use ONLY read_file and write_file.",
        critic_context="Use 'revise' not 'reject' for plans with unnecessary steps.",
    )
    kernel = SumoKernel(settings=settings, domain_context=context)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DomainContext:
    """
    Purely additive domain knowledge injected into agent prompts.

    Fields:
        global_context:    Injected into ALL agents.
        planner_context:   Injected only into the Planner agent.
        critic_context:    Injected only into the Critic agent.
        resolver_context:  Injected only into the Resolver agent.
        synthesis_context: Injected only into the Synthesiser.
    """

    global_context: str = ""
    planner_context: str = ""
    critic_context: str = ""
    resolver_context: str = ""
    synthesis_context: str = ""

    # ── Per-agent context builder ─────────────────────────────────────────────

    def build_for(self, agent_role: str) -> str:
        """
        Return the combined domain context string for a specific agent role.

        Returns an empty string if no context is configured for this role,
        so callers can simply check ``if ctx_str:`` before injecting.

        Args:
            agent_role: One of ``"planner"``, ``"critic"``, ``"resolver"``,
                        ``"synthesis"``.
        """
        role_map = {
            "planner": self.planner_context,
            "critic": self.critic_context,
            "resolver": self.resolver_context,
            "synthesis": self.synthesis_context,
        }
        agent_specific = role_map.get(agent_role, "")

        parts: list[str] = []
        if self.global_context.strip():
            parts.append(self.global_context.strip())
        if agent_specific.strip():
            parts.append(agent_specific.strip())

        if not parts:
            return ""

        return "\n\nPROJECT CONTEXT:\n" + "\n\n".join(parts)

    # ── Auto-detection from workspace ─────────────────────────────────────────

    @classmethod
    async def from_workspace(cls, workspace: str) -> "DomainContext":
        """
        Auto-detect project type and generate appropriate domain context.

        Scans for common project markers:
        - ``pyproject.toml`` / ``requirements.txt`` → Python project details
        - ``package.json`` → Node/React/Next.js
        - ``go.mod`` → Go
        - ``Cargo.toml`` → Rust

        Returns a DomainContext with sensible defaults for the detected stack.
        """
        ws = Path(workspace)
        hints: list[str] = []
        framework: str = ""

        # ── Python detection ──────────────────────────────────────────────
        pyproject = ws / "pyproject.toml"
        if pyproject.exists():
            content = await asyncio.to_thread(pyproject.read_text, "utf-8")
            hints.append("Python project (pyproject.toml found)")

            # Detect frameworks from dependencies
            content_lower = content.lower()
            if "fastapi" in content_lower:
                framework = "FastAPI"
                hints.append("Framework: FastAPI (async, ASGI)")
            elif "django" in content_lower:
                framework = "Django"
                hints.append("Framework: Django (sync, WSGI)")
            elif "flask" in content_lower:
                framework = "Flask"
                hints.append("Framework: Flask")

            # Detect Python version
            if "python_requires" in content or "requires-python" in content:
                import re
                match = re.search(r'(?:python_requires|requires-python)\s*=\s*["\']([^"\']+)', content)
                if match:
                    hints.append(f"Python version: {match.group(1)}")

        requirements = ws / "requirements.txt"
        if requirements.exists() and not pyproject.exists():
            content = await asyncio.to_thread(requirements.read_text, "utf-8")
            hints.append("Python project (requirements.txt found)")
            content_lower = content.lower()
            if "fastapi" in content_lower:
                framework = "FastAPI"
                hints.append("Framework: FastAPI")
            elif "django" in content_lower:
                framework = "Django"
                hints.append("Framework: Django")

        # ── Node.js detection ─────────────────────────────────────────────
        package_json = ws / "package.json"
        if package_json.exists():
            try:
                import json
                content = await asyncio.to_thread(package_json.read_text, "utf-8")
                data = json.loads(content)
                hints.append("Node.js project (package.json found)")
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                if "next" in deps:
                    framework = "Next.js"
                    hints.append("Framework: Next.js")
                elif "react" in deps:
                    framework = "React"
                    hints.append("Framework: React")
                elif "vue" in deps:
                    framework = "Vue.js"
                    hints.append("Framework: Vue.js")
            except Exception:
                hints.append("Node.js project (package.json found, parse failed)")

        # ── Go detection ──────────────────────────────────────────────────
        go_mod = ws / "go.mod"
        if go_mod.exists():
            hints.append("Go project (go.mod found)")

        # ── Rust detection ────────────────────────────────────────────────
        cargo_toml = ws / "Cargo.toml"
        if cargo_toml.exists():
            hints.append("Rust project (Cargo.toml found)")

        if not hints:
            return cls()

        global_ctx = "Detected project stack:\n- " + "\n- ".join(hints)

        # Generate framework-aware planner context
        planner_ctx = ""
        if framework in ("FastAPI", "Django", "Flask"):
            planner_ctx = (
                "This is a Python project.\n"
                "For file editing tasks: use replace_text to modify specific lines. ONLY use write_file if you are creating a completely new file.\n"
                "For code analysis: use ONLY read_file and list_directory.\n"
                "Never use docker, web_search, or fetch_url unless explicitly required.\n"
                "Maximum 5 steps for simple tasks."
            )
        elif framework in ("Next.js", "React", "Vue.js"):
            planner_ctx = (
                "This is a JavaScript/TypeScript project.\n"
                "For file editing tasks: use replace_text to modify specific lines. ONLY use write_file if you are creating a completely new file.\n"
                "For running builds/tests: use shell with npm/yarn commands.\n"
                "Never use docker or fetch_url unless explicitly required.\n"
                "Maximum 5 steps for simple tasks."
            )

        return cls(
            global_context=global_ctx,
            planner_context=planner_ctx,
        )
