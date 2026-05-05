# sumospace/adapters/langchain.py
"""
LangChain Tool Adapter
======================
Wrap any LangChain tool so it runs inside SumoSpace's tool registry.
LangChain is NOT a hard dependency — the import is deferred to call time.

Usage:
    from sumospace.adapters.langchain import wrap_langchain_tool

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    lc_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    sumo_tool = wrap_langchain_tool(lc_tool)

    # Use inside a SumoKernel tool registry:
    kernel.register_tool(sumo_tool)
"""
from __future__ import annotations

from typing import Any

from sumospace.tools import BaseTool, ToolResult


class LangChainToolWrapper(BaseTool):
    """
    Wraps a LangChain BaseTool as a SumoSpace BaseTool.

    The wrapper translates between LangChain's synchronous `run()`
    and SumoSpace's async `run(**kwargs)` interface.
    """

    def __init__(self, lc_tool: Any):
        """
        Args:
            lc_tool: Any LangChain tool with a .name, .description, and .run() method.
        """
        self._lc_tool = lc_tool
        # Mirror LangChain tool metadata
        self.name = getattr(lc_tool, "name", "langchain_tool")
        self.description = getattr(lc_tool, "description", "A wrapped LangChain tool.")

    async def run(self, input: str = "", **kwargs) -> ToolResult:
        """
        Execute the LangChain tool asynchronously.

        SumoSpace passes `input` as the primary query string. Additional
        kwargs are forwarded if the tool supports them.
        """
        import asyncio
        import time

        start = time.monotonic()

        # Build the input string — LangChain tools typically accept a str or dict
        tool_input = input or kwargs.get("query", kwargs.get("tool_input", ""))

        try:
            loop = asyncio.get_event_loop()

            # Prefer arun if available (native async LangChain tools)
            if hasattr(self._lc_tool, "arun"):
                output = await self._lc_tool.arun(tool_input)
            else:
                output = await loop.run_in_executor(
                    None, lambda: self._lc_tool.run(tool_input)
                )

            return ToolResult(
                tool=self.name,
                success=True,
                output=str(output),
                metadata={"lc_tool": self.name, "input": tool_input},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                output="",
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )


def wrap_langchain_tool(lc_tool: Any) -> LangChainToolWrapper:
    """
    Wrap a LangChain tool for use inside SumoSpace.

    Args:
        lc_tool: A LangChain BaseTool instance.

    Returns:
        A SumoSpace-compatible BaseTool wrapper.

    Example:
        from langchain_community.tools import WikipediaQueryRun
        sumo_wiki = wrap_langchain_tool(WikipediaQueryRun(...))
        kernel.register_tool(sumo_wiki)
    """
    # Validate it looks like a LangChain tool
    if not hasattr(lc_tool, "run"):
        raise TypeError(
            f"Expected a LangChain tool with a .run() method, got {type(lc_tool).__name__}. "
            "Make sure langchain or langchain-community is installed: "
            "pip install langchain-community"
        )
    return LangChainToolWrapper(lc_tool)
