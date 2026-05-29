"""
SumoSpace Context Management & Workspace Awareness
==================================================
Replaces unbounded ReAct history with compressed, deterministic Operational Snapshots.
Includes dynamic tool filtering and symbol indexing.
"""

from __future__ import annotations

import os
import ast
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# ─── Environment Awareness ──────────────────────────────────────────────────

@dataclass
class EnvironmentState:
    """Compressed summary of the machine capabilities and constraints."""
    os_name: str = platform.system()
    python_version: str = platform.python_version()
    shell: str = os.environ.get("SHELL", "/bin/sh")
    workspace_root: str = ""
    internet_access: bool = True  # Could be tested/mocked
    gpu_available: bool = False
    
    def format(self) -> str:
        return (
            f"OS: {self.os_name} | Shell: {self.shell} | Python: {self.python_version} | "
            f"Workspace: {self.workspace_root}"
        )


# ─── Symbol Indexing ─────────────────────────────────────────────────────────

class WorkspaceSymbolGraph:
    """
    Maintains a deterministic map of files to their top-level classes and functions.
    Prevents the LLM from having to guess symbol names or read files repeatedly.
    """
    
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root)
        self.index: dict[str, dict[str, list[str]]] = {}

    def update_index(self):
        """Scans all .py files in the workspace and extracts symbols."""
        if not self.workspace.exists():
            return
            
        self.index.clear()
        
        for py_file in self.workspace.rglob("*.py"):
            # Skip hidden dirs or huge venvs for safety
            if any(p.startswith(".") or p in ("venv", "env", "__pycache__") for p in py_file.parts):
                continue
                
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                
                classes = []
                functions = []
                
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append(node.name)
                        
                if classes or functions:
                    rel_path = str(py_file.relative_to(self.workspace))
                    self.index[rel_path] = {
                        "classes": classes,
                        "functions": functions
                    }
            except Exception:
                # Ignore syntax errors or unreadable files during indexing
                pass

    def format_summary(self) -> str:
        """Returns a compressed string representation of the symbol graph."""
        if not self.index:
            return "(No Python symbols indexed)"
            
        lines = []
        for filepath, symbols in list(self.index.items())[:20]:  # Cap at 20 files to prevent bloat
            funcs = symbols["functions"]
            clses = symbols["classes"]
            
            summary_parts = []
            if clses:
                summary_parts.append(f"Classes: {', '.join(clses)}")
            if funcs:
                # Truncate function list if huge
                func_str = ", ".join(funcs[:10]) + ("..." if len(funcs) > 10 else "")
                summary_parts.append(f"Functions: {func_str}")
                
            if summary_parts:
                lines.append(f"- {filepath}: {' | '.join(summary_parts)}")
                
        if len(self.index) > 20:
            lines.append(f"- ... (and {len(self.index) - 20} more files)")
            
        return "\n".join(lines)


# ─── Execution State & Context Manager ──────────────────────────────────────

@dataclass
class StepRecord:
    """A deterministic record of a single step."""
    step_num: int
    tool: str
    thought: str
    success: bool
    output: str
    error: str = ""
    retry_hint: str = ""


class ContextManager:
    """
    Maintains the compressed Operational Snapshot.
    No LLM summarization. Strictly deterministic formatting and truncation.
    """
    
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.env = EnvironmentState(workspace_root=workspace_root)
        self.symbol_graph = WorkspaceSymbolGraph(workspace_root)
        self.history: list[StepRecord] = []
        self.active_files: set[str] = set()
        
    def add_step(self, step: StepRecord):
        self.history.append(step)
        
    def add_active_file(self, filepath: str):
        # Store relative paths for brevity
        try:
            rel = str(Path(filepath).relative_to(self.workspace_root))
            self.active_files.add(rel)
        except ValueError:
            self.active_files.add(str(filepath))
            
    def _get_recent_failures(self, n: int = 3) -> list[StepRecord]:
        return [s for s in self.history if not s.success][-n:]
        
    def _format_completed_actions(self) -> str:
        """Compresses successful history into bullet points without full outputs."""
        successes = [s for s in self.history if s.success]
        if not successes:
            return "- None yet."
            
        lines = []
        for s in successes[-10:]: # keep last 10 successful steps
            # Compress thought to first sentence
            thought_summary = s.thought.split(".")[0].strip() + "." if s.thought else ""
            lines.append(f"- [Step {s.step_num}] {s.tool}: {thought_summary}")
            
        if len(successes) > 10:
            lines.insert(0, f"- ... ({len(successes) - 10} earlier actions truncated)")
            
        return "\n".join(lines)
        
    def _format_recent_outputs(self, n: int = 2, max_chars: int = 800) -> str:
        """Provides raw, but heavily truncated, outputs for the last few steps."""
        if not self.history:
            return "(No recent tool outputs)"
            
        lines = []
        for s in self.history[-n:]:
            status = "SUCCESS" if s.success else "FAILED"
            lines.append(f"--- [Step {s.step_num}: {s.tool}] -> {status} ---")
            
            content = s.output if s.success else s.error
            if len(content) > max_chars:
                lines.append(content[:max_chars] + f"\n... [TRUNCATED: output exceeded {max_chars} chars] ...")
            else:
                lines.append(content)
                
            if not s.success and s.retry_hint:
                lines.append(f"RECOVERY HINT: {s.retry_hint}")
                
            lines.append("")
            
        return "\n".join(lines).strip()

    def filter_tools(self, available_tool_names: list[str]) -> list[str]:
        """
        Heuristic dynamic tool filtering based on context.
        """
        filtered = set(available_tool_names)
        
        # 1. Environment Constraints
        if not self.env.internet_access:
            filtered.discard("web_search")
            filtered.discard("fetch_url")
            
        # 2. Contextual Recommendations
        recent_failures = self._get_recent_failures(2)
        failed_tools = {f.tool for f in recent_failures}
        
        # If replace_text failed recently, ensure AST tools are front-and-center
        if "replace_text" in failed_tools:
            if "replace_function" in available_tool_names:
                filtered.add("replace_function")
            if "replace_class" in available_tool_names:
                filtered.add("replace_class")
            
        # 3. File Context
        has_py_files = any(f.endswith(".py") for f in self.active_files)
        if not has_py_files:
            # If we aren't working on Python, don't spam them with Python AST tools
            filtered.discard("replace_function")
            filtered.discard("replace_class")
            filtered.discard("insert_method")
            filtered.discard("ast_search")
            
        # Ensure critical baseline tools are always present
        baseline = {"read_chunk", "write_file", "search_files", "list_directory", "run_command"}
        filtered.update(t for t in baseline if t in available_tool_names)
            
        return sorted(list(filtered))

    def build_snapshot(self, task: str, available_tool_names: list[str]) -> str:
        """Builds the strictly deterministic Operational Snapshot."""
        
        # Refresh symbol graph implicitly if there are py files active
        if any(f.endswith(".py") for f in self.active_files) or not self.history:
            self.symbol_graph.update_index()
            
        filtered_tools = self.filter_tools(available_tool_names)
        
        snapshot = f"""OPERATIONAL SNAPSHOT
====================
[TASK]
{task}

[ENVIRONMENT]
{self.env.format()}

[ACTIVE FILES]
{', '.join(self.active_files) if self.active_files else '(None)'}

[WORKSPACE SYMBOLS (Python)]
{self.symbol_graph.format_summary()}

[COMPLETED ACTIONS]
{self._format_completed_actions()}

[AVAILABLE TOOLS]
{', '.join(filtered_tools)}
*(Note: Only highly relevant tools are shown. Request other tools if necessary).*

[RECENT TOOL OUTPUTS (Truncated)]
{self._format_recent_outputs(n=3, max_chars=1200)}

Analyze the snapshot and output your next JSON action.
"""
        return snapshot
