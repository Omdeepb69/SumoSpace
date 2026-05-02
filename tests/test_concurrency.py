import pytest
import asyncio
import uuid
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from sumospace.kernel import SumoKernel
from sumospace.settings import SumoSettings
from sumospace.classifier import ClassificationResult, Intent

@pytest.fixture
def mock_provider():
    provider = MagicMock()
    # Mock a simple successful response
    provider.complete.return_value = json.dumps({
        "reasoning": "test",
        "estimated_duration_s": 1,
        "steps": [{"tool": "shell", "description": "ls", "parameters": {"command": "ls"}}]
    })
    provider.initialize = MagicMock()
    return provider

@pytest.mark.asyncio
async def test_concurrency_isolation(tmp_path):
    """Test that multiple kernels with different user_ids don't leak memory/context."""
    workspace = tmp_path
    
    # Mock all slow subsystems
    with patch("sumospace.kernel.ProviderRouter") as MockRouter, \
         patch("sumospace.rag.RAGPipeline") as MockRAG, \
         patch("sumospace.classifier.SumoClassifier") as MockClassifier, \
         patch("sumospace.memory.MemoryManager") as MockMemory, \
         patch("sumospace.ingest.UniversalIngestor") as MockIngestor:
        
        MockRouter.return_value.initialize = AsyncMock()
        MockRouter.return_value.complete = AsyncMock(return_value=json.dumps({
            "reasoning": "test",
            "estimated_duration_s": 1,
            "steps": [{"tool": "shell", "description": "ls", "parameters": {"command": "ls"}}]
        }))
        MockRouter.return_value.stream = AsyncMock()
        
        MockRAG.return_value.initialize = AsyncMock()
        MockClassifier.return_value.initialize = AsyncMock()
        MockMemory.return_value.initialize = AsyncMock()
        MockIngestor.return_value.initialize = AsyncMock()

        async def run_kernel(user_id, task):
            settings = SumoSettings(workspace=str(workspace), user_id=user_id, verbose=False)
            async with SumoKernel(settings=settings) as kernel:
                # Patch classifier
                kernel._classifier.classify = AsyncMock(return_value=ClassificationResult(
                    intent=Intent.GENERAL_QA, confidence=1.0, needs_execution=True, 
                    needs_web=False, needs_retrieval=False
                ))
                
                await kernel.run(task)
                
                recent = kernel._memory.recent(1)
                assert len(recent) >= 1
                return recent[0]["content"]

        # Run two kernels in parallel with different user_ids
        results = await asyncio.gather(
            run_kernel("user1", "task from user1"),
            run_kernel("user2", "task from user2")
        )
        
        assert "user1" in results[0]
        assert "user2" in results[1]
        assert "user2" not in results[0]
        assert "user1" not in results[1]

@pytest.mark.asyncio
async def test_concurrency_shared_safety(tmp_path):
    """Test that multiple kernels with same user_id update shared resources safely."""
    workspace = tmp_path
    user_id = "shared_user"
    
    # Pre-initialize audit index to avoid race on directory creation
    (workspace / ".sumo_audit").mkdir()
    
    with patch("sumospace.kernel.ProviderRouter") as MockRouter, \
         patch("sumospace.rag.RAGPipeline") as MockRAG, \
         patch("sumospace.classifier.SumoClassifier") as MockClassifier, \
         patch("sumospace.memory.MemoryManager") as MockMemory, \
         patch("sumospace.ingest.UniversalIngestor") as MockIngestor:
        
        MockRouter.return_value.initialize = AsyncMock()
        MockRouter.return_value.complete = AsyncMock(return_value=json.dumps({
            "reasoning": "test",
            "estimated_duration_s": 1,
            "steps": [{"tool": "shell", "description": "ls", "parameters": {"command": "ls"}}]
        }))
        MockRouter.return_value.stream = AsyncMock()
        
        MockRAG.return_value.initialize = AsyncMock()
        MockClassifier.return_value.initialize = AsyncMock()
        MockMemory.return_value.initialize = AsyncMock()
        MockIngestor.return_value.initialize = AsyncMock()

        async def run_task(i):
            settings = SumoSettings(workspace=str(workspace), user_id=user_id, verbose=False)
            async with SumoKernel(settings=settings) as kernel:
                # Patch classifier
                kernel._classifier.classify = AsyncMock(return_value=ClassificationResult(
                    intent=Intent.GENERAL_QA, confidence=1.0, needs_execution=True, 
                    needs_web=False, needs_retrieval=False
                ))
                await kernel.run(f"task {i}")

        # Run 10 tasks in parallel
        await asyncio.gather(*[run_task(i) for i in range(10)])
        
        # Check stats index
        index_file = workspace / ".sumo_audit" / "stats_index.json"
        assert index_file.exists()
        with open(index_file, "r") as f:
            stats = json.load(f)
            assert stats["total_sessions"] == 10
