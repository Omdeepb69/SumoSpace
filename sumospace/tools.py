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
    description = "Read the contents of a file."

    async def run(self, path: str, encoding: str = "utf-8", **_) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            content = Path(path).read_text(encoding=encoding, errors="replace")
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
    description = "Write content to a file, creating directories as needed."

    def __init__(self, snapshot_manager=None, run_id: str | None = None):
        self._snapshot_manager = snapshot_manager
        self._run_id = run_id

    async def run(self, path: str, content: str, mode: str = "w", **_) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            p = Path(path)
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
    description = "List files in a directory, optionally filtered by extension."

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
            root = Path(path)
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
    description = "Search for a pattern (regex or literal) in files."

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
            regex = re.compile(pattern)
            results = []
            for root, dirs, files in os.walk(path):
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
                metadata={"pattern": pattern, "matches": len(results)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class PatchFileTool(BaseTool):
    """Apply a unified diff patch to a file."""
    name = "patch_file"
    description = "Apply a code patch to an existing file."

    def __init__(self, snapshot_manager=None, run_id: str | None = None):
        self._snapshot_manager = snapshot_manager
        self._run_id = run_id

    async def run(self, path: str, patch: str, **_) -> ToolResult:
        import time
        start = time.monotonic()
        patch_path = None
        orig_path = None
        try:
            p = Path(path)
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


# ─── Shell Tool ───────────────────────────────────────────────────────────────

class ShellTool(BaseTool):
    name = "shell"
    description = "Run a shell command with timeout. Returns stdout + stderr."

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
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))

    async def close(self):
        if self._browser:
            await self._browser.close()


# ─── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all tools.
    The kernel uses this to look up and execute tools by name.
    """

    def __init__(self, workspace: str = "."):
        self._tools: dict[str, BaseTool] = {}
        self._workspace = workspace
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
        self.register(ReadFileTool())
        self.register(WriteFileTool())
        self.register(ListDirectoryTool())
        self.register(SearchFilesTool())
        self.register(PatchFileTool())
        self.register(ShellTool(workspace=ws))
        self.register(WebSearchTool())
        self.register(FetchURLTool())
        self.register(DockerTool(workspace=ws))
        self.register(DependencyTool(workspace=ws))
        self._discover_plugins()

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]


    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool=name, success=False, output="",
                error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}",
            )
            
        valid, error_msg = tool.validate_params(kwargs)
        if not valid:
            return ToolResult(
                tool=name, success=False, output="",
                error=f"Invalid parameters for '{name}': {error_msg}",
                metadata={"validation_error": True},
            )

        return await tool.run(**kwargs)
