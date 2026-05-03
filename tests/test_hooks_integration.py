import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from sumospace.kernel import SumoKernel
from sumospace.settings import SumoSettings
from sumospace.hooks import HookRegistry
from sumospace.classifier import ClassificationResult, Intent
from sumospace.committee import CommitteeVerdict, ExecutionPlan

@pytest.mark.asyncio
async def test_hook_firing_order(tmp_path):
    fired = []
    hooks = HookRegistry()

    @hooks.on("on_task_start")
    async def h1(task, session_id): fired.append("task_start")

    @hooks.on("on_plan_approved")
    async def h2(plan, verdict): fired.append("plan_approved")

    @hooks.on("on_step_complete")
    async def h3(trace): fired.append(f"step_{trace.step_number}")

    @hooks.on("on_task_complete")
    async def h4(trace): fired.append("task_complete")

    settings = SumoSettings(workspace=str(tmp_path), verbose=False)
    with patch("sumospace.kernel.ProviderRouter") as MockRouter, \
         patch("sumospace.rag.RAGPipeline") as MockRAG, \
         patch("sumospace.classifier.SumoClassifier") as MockClassifier:
        
        MockRouter.return_value.initialize = AsyncMock()
        MockRAG.return_value.initialize = AsyncMock()
        MockClassifier.return_value.initialize = AsyncMock()

        async with SumoKernel(settings=settings) as kernel:
            kernel.hooks = hooks
            
            # Mock classifier and committee to avoid real LLM calls
            kernel._classifier.classify = AsyncMock(return_value=ClassificationResult(
                intent=Intent.GENERAL_QA, confidence=1.0, needs_execution=True, 
                needs_web=False, needs_retrieval=False
            ))
        
            from sumospace.committee import ExecutionStep
            mock_plan = ExecutionPlan(task="test", steps=[
                ExecutionStep(1, "shell", "test", {"command": "echo 1"})
            ])
            kernel._committee.deliberate = AsyncMock(return_value=CommitteeVerdict(
                approved=True, plan=mock_plan, planner_output="mock",
                critic_output="mock", resolver_output="mock"
            ))
            
            await kernel.run("do something")

    assert fired[0] == "task_start"
    assert fired[-1] == "task_complete"
    assert "plan_approved" in fired
    assert "step_1" in fired
    
    # Check order
    assert fired.index("task_start") < fired.index("plan_approved")
    assert fired.index("plan_approved") < fired.index("step_1")
    assert fired.index("step_1") < fired.index("task_complete")
