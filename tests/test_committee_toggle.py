import pytest
from unittest.mock import AsyncMock, MagicMock
from sumospace.settings import SumoSettings
from sumospace.kernel import SumoKernel
from sumospace.committee import CommitteeVerdict
from sumospace.classifier import ClassificationResult, Intent

@pytest.fixture
def mock_kernel():
    settings = SumoSettings(provider="hf", dry_run=True)
    kernel = SumoKernel(settings=settings)
    
    # The kernel's subsystems are None until boot(). Assign mocks directly.
    kernel._initialized = True
    
    kernel._classifier = MagicMock()
    kernel._classifier.classify = AsyncMock(return_value=ClassificationResult(
        intent=Intent.GENERAL_QA,
        confidence=0.9,
        reasoning="test",
        needs_retrieval=False,
        needs_web=False,
        needs_execution=False
    ))
    
    kernel._provider = MagicMock()
    kernel._provider.complete = AsyncMock(return_value="Direct inference answer")
    
    async def mock_stream(*args, **kwargs):
        yield "Direct "
        yield "inference "
        yield "answer"
    kernel._provider.stream = mock_stream

    kernel._committee = MagicMock()
    kernel._committee.deliberate = AsyncMock(return_value=CommitteeVerdict(
        approved=True,
        plan=MagicMock(steps=[]),
    ))
    
    kernel._rag = MagicMock()
    kernel._rag.retrieve = AsyncMock()
    kernel._tools = MagicMock()
    kernel._tools.execute = AsyncMock()
    kernel._memory = MagicMock()
    kernel._memory.recent = MagicMock(return_value=[])
    kernel._memory.add = AsyncMock()
    kernel._memory.context_string = MagicMock(return_value="")
    kernel.hooks = MagicMock()
    kernel.hooks.trigger = AsyncMock()
    kernel._audit_logger = MagicMock()
    kernel._audit_logger.log = MagicMock()
    
    return kernel

@pytest.mark.asyncio
async def test_committee_disabled_returns_direct_inference(mock_kernel):
    mock_kernel.settings.committee_enabled = False
    
    trace = await mock_kernel.run("Hello world")
    
    # Assert committee was bypassed
    mock_kernel._committee.deliberate.assert_not_called()
    
    # Assert direct inference was called
    mock_kernel._provider.complete.assert_called_once()
    
    assert trace.success is True
    assert trace.final_answer == "Direct inference answer"
    assert trace.plan is None

@pytest.mark.asyncio
async def test_committee_disabled_still_fires_hooks(mock_kernel):
    mock_kernel.settings.committee_enabled = False
    trace = await mock_kernel.run("Hello world")
    
    mock_kernel.hooks.trigger.assert_any_call("on_task_start", "Hello world", trace.session_id)
    mock_kernel.hooks.trigger.assert_any_call("on_task_complete", trace)

@pytest.mark.asyncio
async def test_committee_disabled_still_writes_audit_log(mock_kernel):
    mock_kernel.settings.committee_enabled = False
    trace = await mock_kernel.run("Hello world")
    
    mock_kernel._audit_logger.log.assert_called_once()
    call_args = mock_kernel._audit_logger.log.call_args
    assert call_args[0][0] == trace  # first positional arg is the trace
    # verdict should be None (second positional or keyword)
    verdict_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("verdict")
    assert verdict_arg is None

@pytest.mark.asyncio
async def test_plan_only_mode_skips_critic():
    from sumospace.committee import Committee, ExecutionPlan, ExecutionStep
    
    committee = Committee(provider=AsyncMock(), require_consensus=True)
    committee._planner.plan = AsyncMock(return_value=(ExecutionPlan(
        task="test", reasoning="test", estimated_duration_s=1, steps=[ExecutionStep(step_number=1, tool="web_search", description="test", parameters={}, expected_output="test", critical=False)]
    ), "raw"))
    committee._critic.critique = AsyncMock()
    committee._resolver.resolve = AsyncMock()
    
    verdict = await committee.deliberate("test task", mode="plan_only")
    
    assert verdict.approved is True
    assert "plan_only mode" in verdict.rejection_reason
    committee._critic.critique.assert_not_called()
    committee._resolver.resolve.assert_not_called()

@pytest.mark.asyncio
async def test_critique_only_mode_skips_resolver():
    from sumospace.committee import Committee, ExecutionPlan, ExecutionStep
    
    committee = Committee(provider=AsyncMock(), require_consensus=True)
    plan = ExecutionPlan(task="test", reasoning="test", estimated_duration_s=1, steps=[])
    committee._planner.plan = AsyncMock(return_value=(plan, "raw"))
    committee._critic.critique = AsyncMock(return_value=("approve", "looks good", [], [], "raw_critic"))
    committee._resolver.resolve = AsyncMock()
    
    verdict = await committee.deliberate("test task", mode="critique_only")
    
    assert verdict.approved is True
    assert "critique_only mode" in verdict.rejection_reason
    committee._critic.critique.assert_called_once()
    committee._resolver.resolve.assert_not_called()

@pytest.mark.asyncio
async def test_full_mode_runs_all_three_agents():
    from sumospace.committee import Committee, ExecutionPlan, ExecutionStep
    
    committee = Committee(provider=AsyncMock(), require_consensus=True)
    plan = ExecutionPlan(task="test", reasoning="test", estimated_duration_s=1, steps=[])
    committee._planner.plan = AsyncMock(return_value=(plan, "raw_planner"))
    committee._critic.critique = AsyncMock(return_value=("approve", "looks good", [], [], "raw_critic"))
    committee._resolver.resolve = AsyncMock(return_value=(plan, True, "resolved", "raw_resolver"))
    
    verdict = await committee.deliberate("test task", mode="full")
    
    assert verdict.approved is True
    committee._planner.plan.assert_called_once()
    committee._critic.critique.assert_called_once()
    committee._resolver.resolve.assert_called_once()
