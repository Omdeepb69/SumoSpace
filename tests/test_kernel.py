# tests/test_kernel.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sumospace.kernel import SumoKernel, KernelConfig, ExecutionTrace
from sumospace.classifier import Intent
from sumospace.exceptions import KernelBootError, ConsensusFailedError


def fast_config(tmp_path, tmp_chroma) -> KernelConfig:
    """Config that bypasses model loading for tests."""
    return KernelConfig(
        provider="hf",
        model="default",
        workspace=str(tmp_path),
        chroma_path=tmp_chroma,
        dry_run=True,        # No actual execution
        verbose=False,
        require_consensus=False,  # Skip multi-agent for speed
    )


@pytest.mark.asyncio
class TestKernelConfig:
    def test_defaults(self):
        cfg = KernelConfig()
        assert cfg.provider == "hf"
        assert cfg.model == "default"
        assert cfg.embedding_provider == "local"
        assert cfg.dry_run is False
        assert cfg.require_consensus is True
        assert cfg.workspace == "."

    def test_custom_config(self):
        cfg = KernelConfig(
            provider="ollama",
            model="mistral",
            dry_run=True,
            require_consensus=False,
        )
        assert cfg.provider == "ollama"
        assert cfg.model == "mistral"
        assert cfg.dry_run is True


@pytest.mark.asyncio
class TestSumoKernelBoot:
    async def test_boot_with_mock_provider(self, tmp_path, tmp_chroma, mock_provider):
        cfg = fast_config(tmp_path, tmp_chroma)
        kernel = SumoKernel(cfg)

        # Patch provider router to return mock
        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(return_value='{"intent": "general_qa", "confidence": 0.8, "needs_execution": false, "needs_web": false, "needs_retrieval": false, "reasoning": "mock"}')
            MockRouter.return_value = instance

            await kernel.boot()
            assert kernel._initialized is True
            await kernel.shutdown()

    async def test_context_manager(self, tmp_path, tmp_chroma):
        cfg = fast_config(tmp_path, tmp_chroma)
        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(return_value='{"intent": "general_qa", "confidence": 0.8, "needs_execution": false, "needs_web": false, "needs_retrieval": false, "reasoning": "mock"}')
            MockRouter.return_value = instance

            async with SumoKernel(cfg) as kernel:
                assert kernel._initialized is True
            assert kernel._initialized is False


@pytest.mark.asyncio
class TestExecutionTrace:
    def test_trace_defaults(self):
        from sumospace.committee import ExecutionPlan
        trace = ExecutionTrace(
            task="test",
            session_id="abc",
            intent=Intent.GENERAL_QA,
            classification=None,
            plan=None,
        )
        assert trace.success is False
        assert trace.final_answer == ""
        assert trace.step_traces == []
        assert trace.tool_outputs == []
        assert trace.failed_steps == []

    def test_failed_steps_filter(self):
        from sumospace.tools import ToolResult
        from sumospace.kernel import StepTrace
        from sumospace.committee import ExecutionPlan

        trace = ExecutionTrace(
            task="t", session_id="s",
            intent=Intent.GENERAL_QA,
            classification=None, plan=None,
        )
        trace.step_traces = [
            StepTrace(1, "shell", "ok", ToolResult("shell", True, "output"), 10),
            StepTrace(2, "shell", "fail", ToolResult("shell", False, "", "error"), 5),
        ]
        assert len(trace.failed_steps) == 1
        assert trace.failed_steps[0].step_number == 2

    def test_tool_outputs(self):
        from sumospace.tools import ToolResult
        from sumospace.kernel import StepTrace

        trace = ExecutionTrace(
            task="t", session_id="s",
            intent=Intent.GENERAL_QA,
            classification=None, plan=None,
        )
        trace.step_traces = [
            StepTrace(1, "shell", "cmd", ToolResult("shell", True, "out1"), 10),
            StepTrace(2, "read_file", "read", ToolResult("read_file", True, "content"), 5),
        ]
        outputs = trace.tool_outputs
        assert "out1" in outputs
        assert "content" in outputs


@pytest.mark.asyncio
class TestKernelRun:
    async def _make_kernel(self, tmp_path, tmp_chroma, mock_provider):
        cfg = fast_config(tmp_path, tmp_chroma)
        kernel = SumoKernel(cfg)
        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(side_effect=mock_provider.complete)
            instance.provider_name = "mock"
            MockRouter.return_value = instance
            await kernel.boot()
        return kernel

    async def test_run_returns_trace(self, tmp_path, tmp_chroma, mock_provider):
        kernel = SumoKernel(fast_config(tmp_path, tmp_chroma))
        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(side_effect=mock_provider.complete)
            instance.provider_name = "mock"
            MockRouter.return_value = instance

            async with kernel:
                trace = await kernel.run("List Python files")

        assert isinstance(trace, ExecutionTrace)
        assert trace.task == "List Python files"
        assert trace.intent is not None
        assert trace.duration_ms > 0

    async def test_dry_run_no_tool_execution(self, tmp_path, tmp_chroma, mock_provider):
        cfg = fast_config(tmp_path, tmp_chroma)
        cfg.dry_run = True
        kernel = SumoKernel(cfg)

        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(side_effect=mock_provider.complete)
            instance.provider_name = "mock"
            MockRouter.return_value = instance

            async with kernel:
                trace = await kernel.run("Delete all files")

        # Dry run — no steps should have been executed
        assert trace.step_traces == []
        assert "DRY RUN" in trace.final_answer

    async def test_chat_returns_string(self, tmp_path, tmp_chroma, mock_provider):
        kernel = SumoKernel(fast_config(tmp_path, tmp_chroma))
        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(return_value="Mock chat response")
            instance.provider_name = "mock"
            MockRouter.return_value = instance

            async with kernel:
                response = await kernel.chat("Hello, how are you?")

        assert isinstance(response, str)
        assert len(response) > 0

    async def test_ingest_returns_chunk_count(self, tmp_path, tmp_chroma, mock_provider):
        # Create a file to ingest
        (tmp_path / "test.txt").write_text("Some content to ingest for testing. " * 5)
        kernel = SumoKernel(fast_config(tmp_path, tmp_chroma))

        with patch("sumospace.kernel.ProviderRouter") as MockRouter:
            instance = MagicMock()
            instance.initialize = AsyncMock()
            instance.complete = AsyncMock(return_value="")
            instance.provider_name = "mock"
            MockRouter.return_value = instance

            async with kernel:
                count = await kernel.ingest(str(tmp_path / "test.txt"))

        assert isinstance(count, int)
        assert count >= 1
