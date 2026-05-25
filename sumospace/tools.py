# sumospace/tools.py

"""
Tool Registry
=============
All tools available to the kernel's execution planner.
Each tool is a callable with a typed result, execution trace, and error handling.

Tools:
  FileSystem   — read, write, list, search, diff, patch
  Shell        — run commands with timeout + streaming
  Browser      — fetch URL, screenshot, interact (requires sumospace[browser])
  Docker       — build, run, compose, exec
  Dependencies — pip/npm/poetry install, check, export
  WebSearch    — DuckDuckGo (no API key), fallback to direct fetch
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ─── Tool Result ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool: str
    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    recoverable: bool = False
    retry_hint: str = ""


# ─── Base Tool ───────────────────────────────────────────────────────────────

from typing import ClassVar

class BaseTool:
    name: str = "base"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""

    # JSON Schema for parameters — enables validation + LLM-readable docs
    schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    
    # Tool-scoped aliases to map hallucinated parameter names to valid ones
    param_aliases: ClassVar[dict[str, str]] = {}

    # Tool tags for filtering and routing
    tags: ClassVar[list[str]] = []

    async def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def validate_params(self, kwargs: dict) -> tuple[bool, str]:
        """Validate kwargs against schema. Returns (valid, error_message)."""
        try:
            import jsonschema
            jsonschema.validate(kwargs, self.schema)
            return True, ""
        except ImportError:
            return True, ""  # Skip if jsonschema not installed
        except jsonschema.ValidationError as e:
            return False, str(e.message)

    def describe(self) -> dict:
        """Full tool description for LLM prompt construction."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "schema": self.schema,
            "tags": self.tags,
        }


# ─── File System Tools ────────────────────────────────────────────────────────

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file. Parameters: path (string)."
    param_aliases = {"file_path": "path", "filepath": "path", "filename": "path", "file": "path"}

    def __init__(self, workspace: str = "."):
        self._workspace = workspace

    async def run(self, path: str, encoding: str = "utf-8", **_) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            full_path = Path(self._workspace) / path
            content = full_path.read_text(encoding=encoding, errors="replace")
            return ToolResult(
                tool=self.name, success=True,
                output=content,
                metadata={"path": path, "size": len(content)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write text to a new file, or completely overwrite an existing file. Parameters: path (string), content (string)."
    param_aliases = {"file_path": "path", "filepath": "path", "filename": "path", "file": "path", "text": "content", "data": "content", "input": "content", "tool_input": "content", "body": "content", "code": "content", "new_content": "content"}

    def __init__(self, workspace: str = ".", snapshot_manager=None, run_id: str | None = None):
        self._workspace = workspace
        self._snapshot_manager = snapshot_manager
        self._run_id = run_id

    async def run(self, path: str, content: str, mode: str = "w", **_) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            # Strip accidental URL prefixes that LLMs sometimes prepend when using
            # a 'url' field as a content placeholder (e.g. "https://example.com/...\n<actual code>")
            if isinstance(content, str) and content.startswith(("http://", "https://")) and "\n" in content:
                content = content[content.index("\n"):].lstrip("\n")

            # Reject content that looks like just a filename/path rather than real file text
            if isinstance(content, str) and len(content) < 50 and "\n" not in content:
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"Content looks like a filename or placeholder ('{content[:40]}').",
                    recoverable=True,
                    retry_hint="The 'content' parameter must be the COMPLETE text of the file. Do not provide a filename or summary."
                )

            p = Path(self._workspace) / path

            # Execution-Layer Robustness: AST Edit Verification for Python files
            if p.suffix == ".py":
                import ast
                try:
                    ast.parse(content)
                except SyntaxError as syntax_err:
                    return ToolResult(
                        tool=self.name, success=False, output="",
                        error=f"AST SyntaxError in the written content (Line {syntax_err.lineno}): {syntax_err.msg}.",
                        recoverable=True,
                        retry_hint=f"Syntax error at line {syntax_err.lineno}. Check for truncated output, mismatched brackets, or bad indentation."
                    )

            # Snapshot before mutation
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.snapshot_file(self._run_id, str(p.resolve()))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            # Record after-state
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.record_after(self._run_id, str(p.resolve()))
            return ToolResult(
                tool=self.name, success=True,
                output=f"Written {len(content)} chars to {path}",
                metadata={"path": path, "bytes": len(content.encode())},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))



class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List all files and directories in a given path. Parameters: path (string, default '.')."
    param_aliases = {"dir": "path", "directory": "path", "folder": "path"}

    def __init__(self, workspace: str = "."):
        self._workspace = workspace

    async def run(
        self,
        path: str = ".",
        extension: str | None = None,
        recursive: bool = True,
        exclude: list[str] | None = None,
        **_,
    ) -> ToolResult:
        import time
        start = time.monotonic()
        exclude = exclude or ["__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build"]
        try:
            root = Path(self._workspace) / path
            if recursive:
                files = [
                    str(f.relative_to(root))
                    for f in root.rglob("*")
                    if f.is_file()
                    and not any(ex in str(f) for ex in exclude)
                    and (not extension or f.suffix == extension)
                ]
            else:
                files = [
                    str(f.name) for f in root.iterdir()
                    if f.is_file()
                    and (not extension or f.suffix == extension)
                ]
            output = "\n".join(sorted(files))
            return ToolResult(
                tool=self.name, success=True, output=output,
                metadata={"path": path, "count": len(files)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = "Search for a regex pattern across files in a directory. Parameters: pattern (string), path (string, default '.'), extension (string, optional)."
    param_aliases = {"query": "pattern", "search": "pattern", "regex": "pattern", "dir": "path", "directory": "path", "folder": "path"}

    def __init__(self, workspace: str = "."):
        self._workspace = workspace

    async def run(
        self,
        pattern: str,
        path: str = ".",
        extension: str | None = None,
        max_results: int = 50,
        **_,
    ) -> ToolResult:
        import re
        import time
        start = time.monotonic()
        try:
            search_mode = "regex"
            try:
                regex = re.compile(pattern)
            except re.error:
                search_mode = "literal_fallback"
                regex = re.compile(re.escape(pattern))
                
            results = []
            target_path = Path(self._workspace) / path
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "node_modules", ".venv"]]
                for filename in files:
                    if extension and not filename.endswith(extension):
                        continue
                    filepath = Path(root) / filename
                    try:
                        content = filepath.read_text(encoding="utf-8", errors="ignore")
                        for i, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                results.append(f"{filepath}:{i}:  {line.strip()}")
                                if len(results) >= max_results:
                                    break
                    except Exception:
                        continue
                    if len(results) >= max_results:
                        break
            output = "\n".join(results) or "No matches found."
            return ToolResult(
                tool=self.name, success=True, output=output,
                metadata={"pattern": pattern, "matches": len(results), "search_mode": search_mode},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class ReplaceTextTool(BaseTool):
    """Surgical file editing tool that replaces an exact snippet of text."""
    name = "replace_text"
    description = "Replace an exact block of text in a file. Parameters: path (string), old_text (string), new_text (string)."
    param_aliases = {
        "file_path": "path", "filepath": "path", "filename": "path", "file": "path",
        "old": "old_text", "original": "old_text", "original_text": "old_text", "find": "old_text", "target": "old_text",
        "search": "old_text", "search_text": "old_text", "pattern": "old_text", "before": "old_text", "from": "old_text",
        "new": "new_text", "replacement": "new_text", "replacement_text": "new_text", "replace": "new_text",
        "after": "new_text", "replace_with": "new_text", "to": "new_text", "query": "new_text"
    }

    def __init__(self, workspace: str = ".", snapshot_manager=None, run_id: str | None = None):
        self._workspace = workspace
        self._snapshot_manager = snapshot_manager
        self._run_id = run_id

    async def run(self, path: str, old_text: str, new_text: str, **_) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            p = Path(self._workspace) / path
            if not p.exists():
                return ToolResult(tool=self.name, success=False, output="", error=f"File not found: {path}")

            if not old_text:
                return ToolResult(
                    tool=self.name, success=False, output="", 
                    error="old_text cannot be empty.",
                    recoverable=True,
                    retry_hint="Please provide the exact text you want to replace in the 'old_text' parameter."
                )

            # Normalize line endings to prevent brittle LLM serialization failures
            content = p.read_text(encoding="utf-8").replace("\r\n", "\n")
            search_text = old_text.replace("\r\n", "\n")
            replacement = new_text.replace("\r\n", "\n")

            matches = content.count(search_text)
            
            if matches == 0:
                # 2. Flexible Whitespace Match Fallback
                old_lines = [line.strip() for line in search_text.splitlines()]
                while old_lines and not old_lines[0]: old_lines.pop(0)
                while old_lines and not old_lines[-1]: old_lines.pop()

                if not old_lines:
                    return ToolResult(
                        tool=self.name, success=False, output="",
                        error="old_text not found in file (and old_text is empty or just whitespace)."
                    )

                content_lines = content.splitlines()
                content_lines_stripped = [line.strip() for line in content_lines]

                match_start = -1
                matches_found = 0
                for i in range(len(content_lines_stripped) - len(old_lines) + 1):
                    if content_lines_stripped[i:i+len(old_lines)] == old_lines:
                        matches_found += 1
                        match_start = i

                if matches_found == 0:
                    return ToolResult(
                        tool=self.name, success=False, output="",
                        error="old_text not found in file.",
                        recoverable=True,
                        retry_hint="Ensure old_text matches the file exactly. Read the file again if needed."
                    )
                elif matches_found > 1:
                    return ToolResult(
                        tool=self.name, success=False, output="",
                        error=f"old_text matched {matches_found} locations (ignoring whitespace).",
                        recoverable=True,
                        retry_hint="Please provide more surrounding context in old_text to uniquely identify the target."
                    )
                
                # Exactly 1 flexible match!
                match_end = match_start + len(old_lines)
                search_text = "\n".join(content_lines[match_start:match_end])
                
                # Intelligent indentation adjustment
                # If the LLM forgot to indent the replacement block, we add the base indentation back.
                first_line = next((line for line in content_lines[match_start:match_end] if line.strip()), "")
                import re
                base_indent = re.match(r"^[ \t]*", first_line).group(0) if first_line else ""
                
                repl_lines = replacement.split("\n")
                first_repl_line = next((line for line in repl_lines if line.strip()), "")
                repl_indent = re.match(r"^[ \t]*", first_repl_line).group(0) if first_repl_line else ""
                
                # Only adjust if original has indentation and replacement lacks it (or has less)
                if base_indent and len(repl_indent) < len(base_indent):
                    diff_indent = base_indent[len(repl_indent):]
                    adjusted_repl = []
                    for line in repl_lines:
                        if line.strip():
                            adjusted_repl.append(diff_indent + line)
                        else:
                            adjusted_repl.append("")
                    replacement = "\n".join(adjusted_repl)
                
                matches = 1 # Force progression
            
            if matches > 1:
                import re
                # Find line numbers of all matches for actionable diagnostic
                lines = [content[:m.start()].count('\n') + 1 for m in re.finditer(re.escape(search_text), content)]
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"old_text matched {matches} locations (lines {lines}).",
                    recoverable=True,
                    retry_hint="Please provide more surrounding context in old_text to uniquely identify the target."
                )

            # Snapshot before mutation
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.snapshot_file(self._run_id, str(p.resolve()))

            new_content = content.replace(search_text, replacement, 1)

            # Execution-Layer Robustness: AST Edit Verification for Python files
            if p.suffix == ".py":
                import ast
                try:
                    ast.parse(new_content)
                except SyntaxError as syntax_err:
                    return ToolResult(
                        tool=self.name, success=False, output="",
                        error=f"AST SyntaxError in the resulting file after replacement (Line {syntax_err.lineno}): {syntax_err.msg}.",
                        recoverable=True,
                        retry_hint=f"Your replacement introduced a Python syntax error at line {syntax_err.lineno}. Check indentation or syntax."
                    )

            # Atomic write
            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tf:
                tf.write(new_content)
                tmp_path = tf.name
            shutil.move(tmp_path, str(p))

            # Record after-state
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.record_after(self._run_id, str(p.resolve()))

            return ToolResult(
                tool=self.name, success=True,
                output=f"Successfully replaced text in {path}",
                metadata={"path": path, "replaced_chars": len(search_text)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class PatchFileTool(BaseTool):
    """Apply a unified diff patch to a file."""
    name = "patch_file"
    description = "Apply a unified diff patch to a file. Parameters: path (string), patch (string, unified diff format)."

    def __init__(self, workspace: str = ".", snapshot_manager=None, run_id: str | None = None):
        self._workspace = workspace
        self._snapshot_manager = snapshot_manager
        self._run_id = run_id

    async def run(self, path: str, patch: str, **_) -> ToolResult:
        import time
        start = time.monotonic()
        patch_path = None
        orig_path = None
        try:
            p = Path(self._workspace) / path
            # Snapshot before mutation
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.snapshot_file(self._run_id, str(p.resolve()))
            original = p.read_text(encoding="utf-8")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch",
                                            delete=False, encoding="utf-8") as pf:
                pf.write(patch)
                patch_path = pf.name
            with tempfile.NamedTemporaryFile(mode="w", suffix=".orig",
                                            delete=False, encoding="utf-8") as orig_f:
                orig_f.write(original)
                orig_path = orig_f.name

            proc = await asyncio.create_subprocess_exec(
                "patch", orig_path, patch_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                patched = Path(orig_path).read_text(encoding="utf-8")
                p.write_text(patched, encoding="utf-8")
                # Record after-state
                if self._snapshot_manager and self._run_id:
                    self._snapshot_manager.record_after(self._run_id, str(p.resolve()))
                return ToolResult(
                    tool=self.name, success=True,
                    output=f"Patch applied to {path}",
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            else:
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=stderr.decode(),
                )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))
        finally:
            for fp in [patch_path, orig_path]:
                if fp:
                    try:
                        os.unlink(fp)
                    except Exception:
                        pass


# ─── AST-Aware Patching Tool ──────────────────────────────────────────────────

class ReplaceFunctionTool(BaseTool):
    """Symbol-aware function/method replacement using Python AST.
    
    Instead of brittle text matching, this tool parses the file's AST,
    locates a function or method by name, and surgically replaces it.
    """
    name = "replace_function"
    description = (
        "Replace an entire Python function or method body using AST-aware targeting. "
        "Parameters: path (string), function_name (string — use 'ClassName.method' for methods), "
        "new_code (string — the complete new function/method definition including 'def ...:')."
    )
    param_aliases = {
        "file_path": "path", "filepath": "path", "filename": "path", "file": "path",
        "func": "function_name", "name": "function_name", "symbol": "function_name",
        "target": "function_name", "method": "function_name",
        "code": "new_code", "new_body": "new_code", "replacement": "new_code",
        "body": "new_code", "content": "new_code",
    }

    def __init__(self, workspace: str = ".", snapshot_manager=None, run_id: str | None = None):
        self._workspace = workspace
        self._snapshot_manager = snapshot_manager
        self._run_id = run_id

    def _find_symbol(self, tree: 'ast.Module', function_name: str, source_lines: list[str]):
        """Locate a function/method node in the AST.
        
        Supports:
          - 'my_func' — top-level function
          - 'MyClass.my_method' — method inside a class
        
        Returns (node, start_line_0idx, end_line_0idx) or (None, None, None).
        """
        import ast
        
        parts = function_name.split(".", 1)
        
        if len(parts) == 2:
            # Class.method
            class_name, method_name = parts
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                            return self._get_symbol_range(child, source_lines)
            return None, None, None
        else:
            # Top-level function or class
            target_name = parts[0]
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target_name:
                    return self._get_symbol_range(node, source_lines)
            return None, None, None

    def _get_symbol_range(self, node, source_lines: list[str]):
        """Get the full range of a symbol including its decorators and leading whitespace."""
        import ast
        
        # Start from the first decorator, or the def/class line
        if node.decorator_list:
            start_line = min(d.lineno for d in node.decorator_list) - 1  # 0-indexed
        else:
            start_line = node.lineno - 1  # 0-indexed
        
        end_line = node.end_lineno - 1  # 0-indexed, inclusive
        
        return node, start_line, end_line

    async def run(self, path: str, function_name: str, new_code: str, **_) -> ToolResult:
        import ast
        import time
        import re
        start = time.monotonic()
        
        try:
            p = Path(self._workspace) / path
            if not p.exists():
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"File not found: {path}",
                )
            
            if p.suffix != ".py":
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"replace_function only supports Python files (.py), got: {p.suffix}",
                    recoverable=True,
                    retry_hint="Use replace_text for non-Python files.",
                )
            
            if not function_name:
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error="function_name cannot be empty.",
                    recoverable=True,
                    retry_hint="Provide the function name (e.g. 'login_user' or 'MyClass.my_method').",
                )
            
            if not new_code or not new_code.strip():
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error="new_code cannot be empty.",
                    recoverable=True,
                    retry_hint="Provide the complete new function definition including 'def function_name(...):'.",
                )

            content = p.read_text(encoding="utf-8")
            source_lines = content.splitlines(keepends=True)
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"Cannot parse {path}: SyntaxError at line {e.lineno}: {e.msg}",
                    recoverable=True,
                    retry_hint="The file already has a syntax error. Fix it first with replace_text.",
                )
            
            # Find the target symbol
            node, start_idx, end_idx = self._find_symbol(tree, function_name, source_lines)
            
            if node is None:
                # Provide helpful diagnostic
                all_symbols = []
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        all_symbols.append(n.name)
                    elif isinstance(n, ast.ClassDef):
                        all_symbols.append(n.name)
                        for child in ast.iter_child_nodes(n):
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                all_symbols.append(f"{n.name}.{child.name}")
                
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"Symbol '{function_name}' not found in {path}.",
                    recoverable=True,
                    retry_hint=f"Available symbols: {', '.join(all_symbols[:20])}",
                )
            
            # Determine indentation of the original symbol
            original_first_line = source_lines[start_idx] if start_idx < len(source_lines) else ""
            base_indent = re.match(r"^[ \t]*", original_first_line).group(0)
            
            # Normalize the new_code indentation to match the original
            new_lines = new_code.splitlines(keepends=True)
            if new_lines:
                # Detect the indentation of the first non-empty line of new_code
                first_nonempty = next((l for l in new_lines if l.strip()), "")
                new_indent = re.match(r"^[ \t]*", first_nonempty).group(0)
                
                # Re-indent if necessary
                if new_indent != base_indent:
                    adjusted = []
                    for line in new_lines:
                        if line.strip():
                            if line.startswith(new_indent):
                                adjusted.append(base_indent + line[len(new_indent):])
                            else:
                                adjusted.append(base_indent + line.lstrip())
                        else:
                            adjusted.append(line)
                    new_lines = adjusted
                
                # Ensure trailing newline
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] += "\n"
            
            # Build the new file content
            before = source_lines[:start_idx]
            after = source_lines[end_idx + 1:]
            new_content = "".join(before) + "".join(new_lines) + "".join(after)
            
            # AST-validate the result
            try:
                ast.parse(new_content)
            except SyntaxError as syntax_err:
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"Replacement would create SyntaxError at line {syntax_err.lineno}: {syntax_err.msg}.",
                    recoverable=True,
                    retry_hint=f"Your new_code has a syntax error. Check line {syntax_err.lineno}. Make sure you include the full 'def {function_name}(...):' signature.",
                )
            
            # Snapshot before mutation
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.snapshot_file(self._run_id, str(p.resolve()))
            
            # Atomic write
            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tf:
                tf.write(new_content)
                tmp_path = tf.name
            shutil.move(tmp_path, str(p))
            
            # Record after-state
            if self._snapshot_manager and self._run_id:
                self._snapshot_manager.record_after(self._run_id, str(p.resolve()))
            
            lines_removed = end_idx - start_idx + 1
            lines_added = len(new_lines)
            
            return ToolResult(
                tool=self.name, success=True,
                output=f"Replaced '{function_name}' in {path} (removed {lines_removed} lines, added {lines_added} lines)",
                metadata={
                    "path": path,
                    "symbol": function_name,
                    "original_range": f"L{start_idx+1}-L{end_idx+1}",
                    "lines_removed": lines_removed,
                    "lines_added": lines_added,
                },
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Shell Tool ───────────────────────────────────────────────────────────────

class ShellTool(BaseTool):
    name = "shell"
    description = "Run a bash command. Parameters: command (string)."
    param_aliases = {"cmd": "command", "args": "command", "sh": "command"}

    BLOCKED_PATTERNS = [re.compile(p) for p in [
        r"\brm\s+-r", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\{.*\}", r"\bchmod\s+-R\s+777\b"
    ]]

    def __init__(self, workspace: str = ".", timeout: int = 60):
        self.workspace = workspace
        self.timeout = timeout

    async def run(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env_extra: dict | None = None,
        **_,
    ) -> ToolResult:
        import time
        start = time.monotonic()
        timeout = timeout or self.timeout
        cwd = cwd or self.workspace

        # Safety check
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(command):
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"Blocked command pattern match: {pattern.pattern}",
                )

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    tool=self.name, success=False, output="",
                    error=f"Command timed out after {timeout}s",
                )

            output = stdout.decode("utf-8", errors="replace")
            success = proc.returncode == 0
            return ToolResult(
                tool=self.name, success=success, output=output,
                error="" if success else f"Exit code {proc.returncode}",
                metadata={"command": command, "return_code": proc.returncode},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Web Search Tool ─────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    DuckDuckGo search — no API key, no account, free forever.
    Falls back to direct HTTP fetch if DDG is unavailable.
    """
    name = "web_search"
    description = "Search the web using DuckDuckGo (no API key required)."

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    async def run(self, query: str, max_results: int | None = None, **_) -> ToolResult:
        import time
        start = time.monotonic()
        n = max_results or self.max_results
        try:
            import httpx
            # DuckDuckGo Instant Answer API (no key)
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, params=params)
                data = resp.json()

            results = []
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
                if data.get("AbstractURL"):
                    results.append(f"Source: {data['AbstractURL']}")

            for topic in data.get("RelatedTopics", [])[:n]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")
                    if topic.get("FirstURL"):
                        results.append(f"  {topic['FirstURL']}")

            if not results:
                results.append(f"No instant results for: {query}")

            return ToolResult(
                tool=self.name, success=True,
                output="\n".join(results),
                metadata={"query": query, "results": len(results)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class FetchURLTool(BaseTool):
    name = "fetch_url"
    description = "Fetch the text content of a web page."

    async def run(self, url: str, timeout: int = 20, **_) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url)
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type:
                try:
                    from markdownify import markdownify as md
                    text = md(resp.text, heading_style="ATX")
                except ImportError:
                    # Basic HTML strip fallback
                    import re
                    text = re.sub(r"<[^>]+>", "", resp.text)
            else:
                text = resp.text
            return ToolResult(
                tool=self.name, success=True,
                output=text[:10000],  # cap at 10k chars
                metadata={"url": url, "status": resp.status_code},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Docker Tool ─────────────────────────────────────────────────────────────

class DockerTool(BaseTool):
    name = "docker"
    description = "Run Docker CLI commands (build, run, exec, ps, compose)."

    def __init__(self, workspace: str = ".", timeout: int = 120):
        self._shell = ShellTool(workspace=workspace, timeout=timeout)

    async def run(self, command: str, **kwargs) -> ToolResult:
        if not command.startswith("docker"):
            command = f"docker {command}"
        return await self._shell.run(command=command, **kwargs)

    async def build(self, tag: str, context: str = ".", dockerfile: str | None = None) -> ToolResult:
        cmd = f"docker build -t {tag}"
        if dockerfile:
            cmd += f" -f {dockerfile}"
        cmd += f" {context}"
        return await self._shell.run(command=cmd)

    async def run_container(
        self,
        image: str,
        command: str = "",
        ports: dict | None = None,
        env: dict | None = None,
        rm: bool = True,
    ) -> ToolResult:
        parts = ["docker run"]
        if rm:
            parts.append("--rm")
        if ports:
            for host, container in ports.items():
                parts.append(f"-p {host}:{container}")
        if env:
            for k, v in env.items():
                parts.append(f"-e {k}={v}")
        parts.append(shlex.quote(image))
        if command:
            parts.append(command)
        return await self._shell.run(command=" ".join(parts))


# ─── Dependency Management Tool ───────────────────────────────────────────────

class DependencyTool(BaseTool):
    name = "dependencies"
    description = "Install, update, or inspect Python/Node dependencies."

    def __init__(self, workspace: str = "."):
        self._shell = ShellTool(workspace=workspace, timeout=120)

    async def run(self, command: str, **kwargs) -> ToolResult:
        return await self._shell.run(command=command, **kwargs)

    async def pip_install(self, packages: list[str] | str, upgrade: bool = False) -> ToolResult:
        if isinstance(packages, list):
            packages = " ".join(packages)
        cmd = f"pip install {packages}"
        if upgrade:
            cmd += " --upgrade"
        return await self._shell.run(command=cmd)

    async def check_installed(self, package: str) -> ToolResult:
        return await self._shell.run(command=f"pip show {package}")

    async def export_requirements(self, output: str = "requirements.txt") -> ToolResult:
        return await self._shell.run(command=f"pip freeze > {output}")

    async def npm_install(self, packages: list[str] | None = None, dev: bool = False) -> ToolResult:
        if packages:
            flag = "--save-dev" if dev else ""
            cmd = f"npm install {flag} {' '.join(packages)}"
        else:
            cmd = "npm install"
        return await self._shell.run(command=cmd)


# ─── Browser Tool ────────────────────────────────────────────────────────────

class BrowserTool(BaseTool):
    """
    Playwright-based browser automation.
    Requires: pip install sumospace[browser] && playwright install chromium
    """
    name = "browser"
    description = "Automate browser interactions: navigate, click, fill forms, screenshot."

    def __init__(self):
        self._browser = None
        self._page = None

    async def initialize(self):
        try:
            from playwright.async_api import async_playwright
            self._pw = async_playwright()
            pw = await self._pw.__aenter__()
            self._browser = await pw.chromium.launch(headless=True)
            self._page = await self._browser.new_page()
        except ImportError:
            raise ImportError("pip install sumospace[browser] && playwright install chromium")

    async def run(self, url: str, action: str = "fetch", **kwargs) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            if not self._page:
                await self.initialize()
            await self._page.goto(url, timeout=30000)
            if action == "fetch":
                content = await self._page.content()
                try:
                    from markdownify import markdownify as md
                    text = md(content, heading_style="ATX")
                except ImportError:
                    import re
                    text = re.sub(r"<[^>]+>", "", content)
                return ToolResult(
                    tool=self.name, success=True, output=text[:8000],
                    metadata={"url": url, "action": action},
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            elif action == "screenshot":
                path = kwargs.get("output", "screenshot.png")
                await self._page.screenshot(path=path, full_page=True)
                return ToolResult(
                    tool=self.name, success=True, output=f"Screenshot saved: {path}",
                    metadata={"url": url, "action": action, "path": path},
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            else:
                return ToolResult(
                    tool=self.name, success=False, output="", 
                    error=f"Unknown browser action: {action}"
                )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))

    async def close(self):
        if self._browser:
            await self._browser.close()


# ─── Tool Registry ────────────────────────────────────────────────────────────

class InvalidTool(BaseTool):
    """Fallback tool for hallucinated tool names to provide explicit feedback."""
    name = "invalid_tool"
    description = "Internal tool used when the LLM hallucinates a non-existent tool."

    def __init__(self, available_tools: list[str]):
        self._available_tools = available_tools

    async def run(self, hallucinated_tool: str, **_) -> ToolResult:
        return ToolResult(
            tool=self.name,
            success=False,
            output="",
            error=f"Unknown tool: '{hallucinated_tool}'. Available tools: {', '.join(self._available_tools)}"
        )


class ToolRegistry:
    """
    Central registry for all tools.
    The kernel uses this to look up and execute tools by name.
    """

    def __init__(self, workspace: str = ".", snapshot_manager=None):
        self._tools: dict[str, BaseTool] = {}
        self._workspace = workspace
        self._snapshot_manager = snapshot_manager
        self._register_defaults()

    def _discover_plugins(self):
        """Auto-load tools registered via entry points."""
        try:
            from importlib.metadata import entry_points
            from rich.console import Console
            console = Console()
            
            # entry_points(group=...) is Python 3.10+
            try:
                eps = entry_points(group="sumospace.tools")
            except TypeError:
                # Fallback for Python 3.8/3.9
                eps = entry_points().get("sumospace.tools", [])

            for ep in eps:
                try:
                    tool_cls = ep.load()
                    instance = tool_cls()
                    if not isinstance(instance, BaseTool):
                        raise TypeError(f"{tool_cls} is not a BaseTool subclass")
                    self.register(instance)
                    console.print(f"[dim]Plugin loaded: {ep.name} ({ep.value})[/dim]")
                except Exception as e:
                    console.print(
                        f"[yellow]Plugin '{ep.name}' failed to load: {e}[/yellow]"
                    )
        except Exception:
            pass  # importlib.metadata unavailable — skip silently

    def _register_defaults(self):
        ws = self._workspace
        sm = self._snapshot_manager
        self.register(ReadFileTool(workspace=ws))
        self.register(WriteFileTool(workspace=ws, snapshot_manager=sm))
        self.register(ListDirectoryTool(workspace=ws))
        self.register(SearchFilesTool(workspace=ws))
        self.register(ReplaceTextTool(workspace=ws, snapshot_manager=sm))
        self.register(ReplaceFunctionTool(workspace=ws, snapshot_manager=sm))
        self.register(PatchFileTool(workspace=ws, snapshot_manager=sm))
        self.register(ShellTool(workspace=ws))
        self.register(WebSearchTool())
        self.register(FetchURLTool())
        self.register(DockerTool(workspace=ws))
        self.register(DependencyTool(workspace=ws))
        self._discover_plugins()
        # Register the invalid tool placeholder
        self.register(InvalidTool(available_tools=list(self._tools.keys())))

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, any]]:
        return [{"name": t.name, "description": t.description, "schema": getattr(t, "schema", {})} for t in self._tools.values()]


    async def execute(self, name: str, run_id: str | None = None, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool=name, success=False, output="",
                error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}",
            )

        if hasattr(tool, "_run_id"):
            tool._run_id = run_id

        # Normalize LLM parameter hallucinations based on tool-scoped aliases
        normalized = {}
        for k, v in kwargs.items():
            normalized[tool.param_aliases.get(k, k)] = v
        kwargs = normalized

        valid, error_msg = tool.validate_params(kwargs)
        if not valid:
            return ToolResult(
                tool=name, success=False, output="",
                error=f"Invalid parameters for '{name}': {error_msg}",
                metadata={"validation_error": True},
            )
        try:
            return await tool.run(**kwargs)
        except TypeError as e:
            return ToolResult(
                tool=name, success=False, output="",
                error=f"Parameter mismatch for '{name}': {e}",
            )
        except Exception as e:
            return ToolResult(
                tool=name, success=False, output="",
                error=f"Execution error in '{name}': {e}",
            )
