# examples/desktop_agent/agent.py

"""
SumoSpace Desktop Agent
========================
An autonomous agent that controls the desktop via screenshots + keyboard/mouse.
Uses SumoSpace's ProviderRouter as the AI brain, MemoryManager for history,
ToolRegistry for action execution, and ScopeManager for session isolation.

Usage:
    # CLI mode
    python examples/desktop_agent/agent.py "Open VS Code and create a new Python file"

    # Programmatic
    agent = DesktopAgent(provider="ollama", model="phi3:mini")
    await agent.initialize()
    result = await agent.execute("Open the file manager")
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure sumospace is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sumospace.memory import MemoryManager
from sumospace.providers import ProviderRouter
from sumospace.scope import ScopeManager
from sumospace.tools import ToolRegistry, ToolResult

from desktop_tools import register_desktop_tools

console = Console()


# ─── Agent Config ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for the Desktop Agent."""
    provider: str = "ollama"          # hf | ollama | gemini | openai | anthropic
    model: str = "default"            # model name or alias
    max_steps: int = 15               # safety limit per task
    screenshot_dir: str = "/tmp/sumo_screenshots"
    user_id: str = "desktop_agent"    # scope isolation
    chroma_base: str = ".sumo_desktop_db"
    verbose: bool = True
    step_delay: float = 1.0           # seconds between steps
    use_vision: bool = False          # if True, encode screenshot as base64 for vision models


# ─── Action Schema ────────────────────────────────────────────────────────────

@dataclass
class AgentAction:
    """A single planned action from the LLM."""
    tool: str
    parameters: dict[str, Any]
    reasoning: str = ""
    is_done: bool = False


@dataclass
class StepResult:
    """Record of one agent step."""
    step: int
    action: AgentAction
    tool_result: ToolResult
    screenshot_path: str = ""
    screen_text: str = ""
    duration_ms: float = 0.0


@dataclass
class TaskResult:
    """Final result of executing a task."""
    task: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    summary: str = ""
    total_duration_ms: float = 0.0
    session_id: str = ""


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Desktop Automation Agent. You control a Linux desktop by executing ONE tool per turn.

AVAILABLE TOOLS:
- take_screenshot: Capture the screen. Parameters: {} (none needed)
- read_screen: OCR the screen to get all visible text. Parameters: {} (none needed)
- click_at: Left-click. Parameters: {"x": int, "y": int}
- double_click: Double-click. Parameters: {"x": int, "y": int}
- right_click: Right-click. Parameters: {"x": int, "y": int}
- type_text: Type text via keyboard. Parameters: {"text": "string"}
- hotkey: Keyboard shortcut. Parameters: {"keys": "ctrl,s"} (comma-separated)
- press_key: Single key press. Parameters: {"key": "enter"}
- move_mouse: Move cursor. Parameters: {"x": int, "y": int}
- scroll: Scroll wheel. Parameters: {"clicks": int} (positive=up, negative=down)
- drag_to: Click-drag to position. Parameters: {"x": int, "y": int}
- open_app: Launch application. Parameters: {"app": "code", "args": "/optional/path"}
- wait: Pause execution. Parameters: {"seconds": float}
- shell: Run a shell command. Parameters: {"command": "ls -la"}

RESPONSE FORMAT — respond with ONLY this JSON, nothing else:
{
  "reasoning": "What I observe and why I'm choosing this action",
  "tool": "tool_name",
  "parameters": {},
  "is_done": false
}

CRITICAL RULES:
1. OBSERVE FIRST: Your first action should always be read_screen to see what's on the desktop.
2. ONE ACTION PER TURN: Execute exactly one tool, then you'll see the result.
3. NEVER REPEAT: If you already opened an app, do NOT open it again. Interact with it using hotkeys or clicks.
4. VERIFY AFTER ACTING: After a significant action (open_app, click, type), observe the result on the next turn.
5. OPEN APP WITH ARGS: To open a specific folder or file in VS Code, use open_app with args (e.g. {"app": "code", "args": "/path/to/folder"}). Do not just open the app and try to click through menus.
6. USE HOTKEYS: To create a new file in VS Code use hotkey ctrl+n. To save use ctrl+s. To open terminal use ctrl+`.
7. FINISH: When the full task is complete, set "is_done": true, "tool": "none", and explain what was accomplished.
8. If an app is already visible on screen, interact with it directly — do NOT open it again.

WORKFLOW EXAMPLE for "Open VS Code and open the sumospace project":
  Step 1: read_screen → see what's on screen
  Step 2: open_app {"app": "code", "args": "/mnt/data/projects/sumospace/sumospace"} → launch VS Code directly in the project
  Step 3: wait {"seconds": 3} → let it load
  Step 4: read_screen → verify VS Code is open with the project
  Step 5: is_done → task complete"""


# ─── Desktop Agent ────────────────────────────────────────────────────────────

class DesktopAgent:
    """
    Autonomous desktop automation agent powered by SumoSpace.

    Architecture:
        ProviderRouter  → LLM brain (plans next action)
        ToolRegistry    → Desktop tools + built-in SumoSpace tools
        MemoryManager   → Conversation history (persisted via ChromaDB)
        ScopeManager    → Per-session filesystem isolation
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._provider: ProviderRouter | None = None
        self._tools: ToolRegistry | None = None
        self._memory: MemoryManager | None = None
        self._scope: ScopeManager | None = None
        self._initialized = False
        self._session_id = uuid.uuid4().hex[:12]

    async def initialize(self):
        """Boot all SumoSpace subsystems."""
        if self._initialized:
            return

        cfg = self.config

        if cfg.verbose:
            console.print(Panel(
                f"[bold cyan]SumoSpace Desktop Agent[/bold cyan]\n"
                f"Provider: [green]{cfg.provider}[/green]  "
                f"Model: [green]{cfg.model}[/green]\n"
                f"Max steps: [yellow]{cfg.max_steps}[/yellow]  "
                f"Session: [dim]{self._session_id}[/dim]",
                title="🖥️  Desktop Agent",
                border_style="cyan",
            ))

        # 1. Provider (AI brain)
        self._provider = ProviderRouter(
            provider=cfg.provider,
            model=cfg.model if cfg.model != "default" else None,
        )
        await self._provider.initialize()

        # 2. Scope Manager (session isolation)
        self._scope = ScopeManager(
            chroma_base=cfg.chroma_base,
            level="session",
        )

        # 3. Memory (persistent conversation history)
        resolved_path = self._scope.resolve(
            user_id=cfg.user_id,
            session_id=self._session_id,
        )
        self._memory = MemoryManager(
            chroma_path=resolved_path,
            scope_manager=self._scope,
            user_id=cfg.user_id,
            session_id=self._session_id,
        )
        await self._memory.initialize()

        # 4. Tool Registry (built-in + desktop tools)
        self._tools = ToolRegistry(workspace=".")
        desktop_names = register_desktop_tools(self._tools)

        if cfg.verbose:
            console.print(
                f"[green]✓ Initialized[/green] — "
                f"{len(desktop_names)} desktop tools registered"
            )

        self._initialized = True

    async def execute(self, task: str, on_step=None) -> TaskResult:
        """
        Execute a desktop task autonomously.

        Uses an observe-act-verify loop with deduplication:
          1. Observe screen (auto OCR)
          2. Ask LLM for next action
          3. Block duplicate/repeated actions
          4. Execute action via ToolRegistry
          5. Auto-observe after significant actions
          6. Repeat until done or max_steps
        
        Args:
            task: The natural language task.
            on_step: Optional callback func(StepResult) invoked after each step.
        """
        if not self._initialized:
            await self.initialize()

        cfg = self.config
        start = time.monotonic()
        steps: list[StepResult] = []

        if cfg.verbose:
            console.print(f"\n[bold]Task:[/bold] {task}\n")

        # Store task in memory
        await self._memory.add("user", f"Desktop task: {task}")

        # State tracking
        action_history: list[str] = []
        last_screen_text: str = ""
        completed_subtasks: list[str] = []
        action_counter: Counter = Counter()  # track how many times each action is used
        last_action_key: str = ""
        consecutive_repeats: int = 0

        for step_num in range(1, cfg.max_steps + 1):
            step_start = time.monotonic()

            if cfg.verbose:
                console.print(f"[cyan]━━━ Step {step_num}/{cfg.max_steps} ━━━[/cyan]")

            # Build context with rich state info
            user_prompt = self._build_prompt(
                task=task,
                history=action_history,
                step_num=step_num,
                screen_text=last_screen_text,
                completed=completed_subtasks,
            )

            # Ask LLM for next action
            action = await self._get_next_action(user_prompt)

            # ── Deduplication: block repeated actions ────────────────────
            action_key = f"{action.tool}:{json.dumps(action.parameters, sort_keys=True)}"

            if action_key == last_action_key:
                consecutive_repeats += 1
            else:
                consecutive_repeats = 0

            # If same action repeated 2+ times, force observation or mark done
            if consecutive_repeats >= 2:
                if cfg.verbose:
                    console.print(
                        f"  [yellow]⚠ Blocked repeat action ({action.tool}), "
                        f"forcing observation[/yellow]"
                    )
                # Force a read_screen to break the loop
                action = AgentAction(
                    tool="read_screen",
                    parameters={},
                    reasoning="Repeated action blocked — observing screen to reassess",
                )
                consecutive_repeats = 0

            # Block opening same app more than once
            if action.tool == "open_app":
                app_name = action.parameters.get("app", "")
                open_count = sum(
                    1 for h in action_history
                    if "open_app" in h and app_name.lower() in h.lower() and "FAILED" not in h
                )
                if open_count >= 1:
                    if cfg.verbose:
                        console.print(
                            f"  [yellow]⚠ '{app_name}' already opened previously, "
                            f"skipping to next action[/yellow]"
                        )
                    # Force read_screen so it sees the app is already open
                    action = AgentAction(
                        tool="read_screen",
                        parameters={},
                        reasoning=f"Agent tried to open '{app_name}' again, but it was already opened. Observing screen instead to interact with it.",
                    )

            last_action_key = f"{action.tool}:{json.dumps(action.parameters, sort_keys=True)}"
            action_counter[action.tool] += 1

            # ── Handle is_done ───────────────────────────────────────────
            if action.is_done:
                if cfg.verbose:
                    console.print(f"[green]✓ Task complete:[/green] {action.reasoning}")
                
                final_step = StepResult(
                    step=step_num,
                    action=action,
                    tool_result=ToolResult(tool="none", success=True, output="Task complete"),
                    duration_ms=(time.monotonic() - step_start) * 1000,
                )
                steps.append(final_step)
                completed_subtasks.append(action.reasoning)
                
                if on_step:
                    if asyncio.iscoroutinefunction(on_step):
                        await on_step(final_step)
                    else:
                        on_step(final_step)
                        
                break

            if cfg.verbose:
                console.print(f"  [dim]Think:[/dim] {action.reasoning}")
                console.print(
                    f"  [yellow]Action:[/yellow] {action.tool}"
                    f"({json.dumps(action.parameters)[:100]})"
                )

            # ── Execute action via ToolRegistry ──────────────────────────
            result = await self._tools.execute(action.tool, **action.parameters)

            if cfg.verbose:
                if result.success:
                    preview = result.output[:120].replace("\n", " ")
                    console.print(f"  [green]✓[/green] {preview}")
                else:
                    console.print(f"  [red]✗[/red] {result.error}")

            # Record step
            step_result = StepResult(
                step=step_num,
                action=action,
                tool_result=result,
                duration_ms=(time.monotonic() - step_start) * 1000,
            )

            # Capture screenshot path and screen text
            if action.tool == "take_screenshot" and result.success:
                step_result.screenshot_path = result.metadata.get("path", "")

            if action.tool == "read_screen" and result.success:
                step_result.screen_text = result.output
                last_screen_text = result.output

            steps.append(step_result)

            if on_step:
                if asyncio.iscoroutinefunction(on_step):
                    await on_step(step_result)
                else:
                    on_step(step_result)

            # Update action history with result
            status = "OK" if result.success else "FAILED"
            output_preview = result.output[:120] if result.success else result.error[:120]
            action_summary = (
                f"Step {step_num}: [{action.tool}] "
                f"{json.dumps(action.parameters)[:60]} → "
                f"{status}: {output_preview}"
            )
            action_history.append(action_summary)

            # Track completed subtasks
            if action.tool == "open_app" and result.success:
                app = action.parameters.get("app", "unknown")
                completed_subtasks.append(f"Opened {app}")
            elif action.tool in ("type_text", "hotkey") and result.success:
                completed_subtasks.append(f"{action.tool}: {result.output[:50]}")

            # Store in memory
            await self._memory.add(
                "assistant",
                f"Action: {action.tool} | Result: {result.output[:200]}",
            )

            # ── Auto-observe after significant actions ───────────────────
            # After open_app or wait, automatically take a screenshot + OCR
            if action.tool in ("open_app", "wait") and result.success:
                if cfg.verbose:
                    console.print("  [dim]↳ Auto-observing screen...[/dim]")
                await asyncio.sleep(1.0)
                ocr_result = await self._tools.execute("read_screen")
                if ocr_result.success:
                    last_screen_text = ocr_result.output
                    action_history.append(
                        f"  ↳ Auto-OCR: screen shows {len(ocr_result.output)} chars"
                    )

            # Delay between steps
            if cfg.step_delay > 0:
                await asyncio.sleep(cfg.step_delay)

        # ── Auto-finish if max steps reached ─────────────────────────────
        if not any(s.action.is_done for s in steps):
            # Check if meaningful work was done
            meaningful_actions = [
                s for s in steps
                if s.action.tool not in ("take_screenshot", "read_screen", "wait")
                and s.tool_result.success
            ]
            if meaningful_actions:
                if cfg.verbose:
                    console.print(
                        "[yellow]⚠ Max steps reached, but meaningful "
                        "actions were taken[/yellow]"
                    )

        # Generate summary
        total_ms = (time.monotonic() - start) * 1000
        success = any(s.action.is_done for s in steps) or len(completed_subtasks) > 0
        summary = await self._summarize(task, steps, completed_subtasks)

        if cfg.verbose:
            console.print(f"\n[bold]{'✓' if success else '✗'} Summary:[/bold] {summary}")
            console.print(
                f"[dim]{len(steps)} steps, {total_ms:.0f}ms total[/dim]\n"
            )

        return TaskResult(
            task=task,
            success=success,
            steps=steps,
            summary=summary,
            total_duration_ms=total_ms,
            session_id=self._session_id,
        )

    # ── Internal Methods ─────────────────────────────────────────────────────

    def _build_prompt(
        self,
        task: str,
        history: list[str],
        step_num: int,
        screen_text: str,
        completed: list[str],
    ) -> str:
        """Build the user prompt with full state context."""
        parts = [f"TASK: {task}"]

        if completed:
            parts.append(f"\nCOMPLETED SO FAR:\n" + "\n".join(f"  ✓ {c}" for c in completed))

        parts.append(f"\nCURRENT STEP: {step_num}")

        if screen_text:
            # Truncate screen text to keep prompt manageable
            truncated = screen_text[:1500]
            parts.append(f"\nCURRENT SCREEN TEXT (via OCR):\n{truncated}")

        if history:
            # Show recent history
            recent = history[-8:]
            parts.append(f"\nACTION HISTORY (recent):\n" + "\n".join(recent))

        parts.append(
            "\nWhat is the NEXT SINGLE action to make progress? "
            "Do NOT repeat any action from the history. "
            "If the task is fully complete, set is_done=true. "
            "Respond with ONLY JSON."
        )
        return "\n".join(parts)

    async def _get_next_action(self, user_prompt: str) -> AgentAction:
        """Call the LLM to decide the next action."""
        import re

        try:
            raw = await self._provider.complete(
                user=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=512,
            )

            # Strip markdown fences if present
            raw = re.sub(r"```json\s*|```\s*", "", raw).strip()

            # Extract JSON from response (handle extra text around it)
            start_idx = raw.find("{")
            end_idx = raw.rfind("}")
            if start_idx != -1 and end_idx != -1:
                raw = raw[start_idx:end_idx + 1]

            data = json.loads(raw)

            tool = data.get("tool", "read_screen")
            # Normalize "none" tool for is_done
            if tool.lower() in ("none", "null", "done", "finish", "complete"):
                return AgentAction(
                    tool="none",
                    parameters={},
                    reasoning=data.get("reasoning", "Task complete"),
                    is_done=True,
                )

            return AgentAction(
                tool=tool,
                parameters=data.get("parameters", {}),
                reasoning=data.get("reasoning", ""),
                is_done=bool(data.get("is_done", False)),
            )
        except Exception as e:
            # If parsing fails, observe screen to get back on track
            return AgentAction(
                tool="read_screen",
                parameters={},
                reasoning=f"LLM response parse failed ({e}), observing screen to reassess",
            )

    async def _summarize(
        self, task: str, steps: list[StepResult], completed: list[str]
    ) -> str:
        """Generate a summary of what was accomplished."""
        if completed:
            completed_str = ", ".join(completed)
        else:
            completed_str = "No significant actions completed"

        actions = "\n".join(
            f"  {s.step}. {s.action.tool}: {s.tool_result.output[:60]}"
            for s in steps
            if s.action.tool not in ("take_screenshot", "read_screen")
        )
        try:
            return await self._provider.complete(
                user=(
                    f"Task: {task}\n"
                    f"Completed subtasks: {completed_str}\n"
                    f"Actions:\n{actions}\n\n"
                    "Write a 1-2 sentence summary of what was accomplished."
                ),
                system="You are a concise task summarizer.",
                temperature=0.2,
                max_tokens=128,
            )
        except Exception:
            return f"Completed: {completed_str}"

    async def cleanup(self):
        """Clean up session data."""
        if self._scope and self.config.user_id:
            stats = self._scope.get_stats(self.config.user_id)
            if self.config.verbose:
                console.print(
                    f"[dim]Session {self._session_id}: "
                    f"{stats.total_disk_mb:.1f}MB used[/dim]"
                )

    async def list_sessions(self) -> list:
        """List all past agent sessions."""
        if self._scope:
            return self._scope.list_sessions(self.config.user_id)
        return []


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SumoSpace Desktop Agent — autonomous desktop automation"
    )
    parser.add_argument("task", help="Task to execute (natural language)")
    parser.add_argument("--provider", "-p", default="ollama",
                        help="LLM provider: ollama | hf | gemini | openai | anthropic")
    parser.add_argument("--model", "-m", default="default",
                        help="Model name or alias")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum steps per task")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    config = AgentConfig(
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    agent = DesktopAgent(config)
    try:
        await agent.initialize()
        result = await agent.execute(args.task)

        # Print results table
        table = Table(title="Execution Summary")
        table.add_column("Step", style="cyan")
        table.add_column("Tool", style="yellow")
        table.add_column("Result", style="green")
        table.add_column("Time", style="dim")

        for s in result.steps:
            status = "✓" if s.tool_result.success else "✗"
            table.add_row(
                str(s.step),
                s.action.tool,
                f"{status} {s.tool_result.output[:60]}",
                f"{s.duration_ms:.0f}ms",
            )

        console.print(table)

        if not result.success:
            sys.exit(1)

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
