# tests/test_langchain_adapter.py
"""Tests for LangChain tool adapter."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from sumospace.adapters.langchain import wrap_langchain_tool, LangChainToolWrapper
from sumospace.tools import ToolResult


def make_lc_tool(name="wiki", description="Wikipedia search", run_return="Paris is the capital of France."):
    """Create a minimal mock LangChain tool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.run = MagicMock(return_value=run_return)
    # No arun by default — tests sync path
    del tool.arun
    return tool


def test_wrap_langchain_tool_returns_wrapper():
    lc_tool = make_lc_tool()
    wrapped = wrap_langchain_tool(lc_tool)
    assert isinstance(wrapped, LangChainToolWrapper)
    assert wrapped.name == "wiki"
    assert wrapped.description == "Wikipedia search"


def test_wrap_langchain_tool_rejects_non_tool():
    with pytest.raises(TypeError, match=".run\\(\\)"):
        wrap_langchain_tool({"not": "a tool"})


@pytest.mark.asyncio
async def test_wrapper_run_sync_tool():
    lc_tool = make_lc_tool(run_return="Paris is the capital.")
    wrapped = wrap_langchain_tool(lc_tool)

    result = await wrapped.run(input="capital of France")

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert "Paris" in result.output
    assert result.tool == "wiki"
    lc_tool.run.assert_called_once_with("capital of France")


@pytest.mark.asyncio
async def test_wrapper_run_async_tool():
    """If the LangChain tool has arun(), it should be preferred."""
    lc_tool = make_lc_tool()
    lc_tool.arun = AsyncMock(return_value="Async result from LangChain")

    wrapped = wrap_langchain_tool(lc_tool)
    result = await wrapped.run(input="query")

    assert result.success is True
    assert "Async result" in result.output
    lc_tool.arun.assert_called_once_with("query")


@pytest.mark.asyncio
async def test_wrapper_run_failure_returns_error_result():
    lc_tool = make_lc_tool()
    lc_tool.run = MagicMock(side_effect=RuntimeError("API rate limit exceeded"))
    # Remove arun to force sync path
    if hasattr(lc_tool, "arun"):
        del lc_tool.arun

    wrapped = wrap_langchain_tool(lc_tool)
    result = await wrapped.run(input="anything")

    assert result.success is False
    assert "rate limit" in result.error


@pytest.mark.asyncio
async def test_wrapper_forwards_kwargs_as_input():
    """Fallback: if no 'input' kwarg, use 'query' kwarg."""
    lc_tool = make_lc_tool(run_return="result")
    wrapped = wrap_langchain_tool(lc_tool)

    result = await wrapped.run(query="what is python?")
    assert result.success is True
    lc_tool.run.assert_called_once_with("what is python?")
