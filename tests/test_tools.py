# tests/test_tools.py

import pytest
from pathlib import Path
from sumospace.tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    SearchFilesTool,
    ShellTool,
    WebSearchTool,
    FetchURLTool,
    ToolRegistry,
    ToolResult,
)


class TestReadFileTool:
    def setup_method(self):
        self.tool = ReadFileTool()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, World!")
        result = await self.tool.run(path=str(f))
        assert result.success is True
        assert result.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        result = await self.tool.run(path="/nonexistent/file.txt")
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_read_unicode_file(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("日本語テスト — Arabic: مرحبا")
        result = await self.tool.run(path=str(f))
        assert result.success is True
        assert "日本語" in result.output

    @pytest.mark.asyncio
    async def test_metadata_populated(self, tmp_path):
        f = tmp_path / "data.txt"
        content = "some content here"
        f.write_text(content)
        result = await self.tool.run(path=str(f))
        assert result.metadata["path"] == str(f)
        assert result.metadata["size"] == len(content)


class TestWriteFileTool:
    def setup_method(self):
        self.tool = WriteFileTool()

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "output.txt"
        result = await self.tool.run(path=str(f), content="Written content")
        assert result.success is True
        assert f.read_text() == "Written content"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, tmp_path):
        f = tmp_path / "deep" / "nested" / "file.txt"
        result = await self.tool.run(path=str(f), content="nested")
        assert result.success is True
        assert f.exists()
        assert f.read_text() == "nested"

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        result = await self.tool.run(path=str(f), content="new content")
        assert result.success is True
        assert f.read_text() == "new content"


class TestListDirectoryTool:
    def setup_method(self):
        self.tool = ListDirectoryTool()

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_workspace):
        result = await self.tool.run(path=str(tmp_workspace))
        assert result.success is True
        assert "main.py" in result.output
        assert "README.md" in result.output

    @pytest.mark.asyncio
    async def test_filter_by_extension(self, tmp_workspace):
        result = await self.tool.run(path=str(tmp_workspace), extension=".py")
        assert result.success is True
        assert ".py" in result.output
        assert "README.md" not in result.output

    @pytest.mark.asyncio
    async def test_nonrecursive_list(self, tmp_workspace):
        result = await self.tool.run(path=str(tmp_workspace), recursive=False)
        assert result.success is True
        # Only top-level files
        assert "README.md" in result.output

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self):
        result = await self.tool.run(path="/nonexistent/path")
        # os.walk silently returns empty for missing dirs — success with 0 files
        assert result.success is True
        assert result.metadata["count"] == 0

    @pytest.mark.asyncio
    async def test_metadata_count(self, tmp_workspace):
        result = await self.tool.run(path=str(tmp_workspace))
        assert result.metadata["count"] >= 3


class TestSearchFilesTool:
    def setup_method(self):
        self.tool = SearchFilesTool()

    @pytest.mark.asyncio
    async def test_search_finds_match(self, tmp_workspace):
        result = await self.tool.run(
            pattern="def hello",
            path=str(tmp_workspace),
        )
        assert result.success is True
        assert "hello" in result.output
        assert "main.py" in result.output

    @pytest.mark.asyncio
    async def test_search_no_match(self, tmp_workspace):
        result = await self.tool.run(
            pattern="ZZZNOMATCH999",
            path=str(tmp_workspace),
        )
        assert result.success is True
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_search_with_extension_filter(self, tmp_workspace):
        result = await self.tool.run(
            pattern="import",
            path=str(tmp_workspace),
            extension=".py",
        )
        assert result.success is True
        assert "README.md" not in result.output

    @pytest.mark.asyncio
    async def test_search_regex_pattern(self, tmp_workspace):
        result = await self.tool.run(
            pattern=r"def \w+\(",
            path=str(tmp_workspace),
        )
        assert result.success is True
        assert "def " in result.output


class TestShellTool:
    def setup_method(self):
        self.tool = ShellTool()

    @pytest.mark.asyncio
    async def test_run_echo(self):
        result = await self.tool.run(command="echo hello_world")
        assert result.success is True
        assert "hello_world" in result.output

    @pytest.mark.asyncio
    async def test_run_pwd(self, tmp_path):
        result = await self.tool.run(command="pwd", cwd=str(tmp_path))
        assert result.success is True
        assert str(tmp_path) in result.output.strip()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        result = await self.tool.run(command="exit 1")
        assert result.success is False
        assert "Exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        result = await self.tool.run(command="sleep 10", timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        result = await self.tool.run(command="rm -rf /")
        assert result.success is False
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_metadata_populated(self):
        result = await self.tool.run(command="echo test")
        assert "command" in result.metadata
        assert "return_code" in result.metadata

    @pytest.mark.asyncio
    async def test_multiline_output(self):
        result = await self.tool.run(command="printf 'line1\\nline2\\nline3'")
        assert result.success is True
        assert "line1" in result.output
        assert "line3" in result.output


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_list_tools(self):
        tools = self.registry.list_tools()
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "write_file" in names
        assert "shell" in names
        assert "web_search" in names
        assert "list_directory" in names

    def test_get_existing_tool(self):
        tool = self.registry.get("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_nonexistent_tool(self):
        tool = self.registry.get("nonexistent_tool")
        assert tool is None

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        result = await self.registry.execute("no_such_tool")
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_shell(self):
        result = await self.registry.execute("shell", command="echo registry_test")
        assert result.success is True
        assert "registry_test" in result.output

    def test_register_custom_tool(self):
        from sumospace.tools import BaseTool

        class MyTool(BaseTool):
            name = "my_custom_tool"
            description = "A custom test tool"

            async def run(self, **kwargs) -> ToolResult:
                return ToolResult(tool=self.name, success=True, output="custom output")

        self.registry.register(MyTool())
        assert self.registry.get("my_custom_tool") is not None
