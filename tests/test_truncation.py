import pytest
from sumospace.kernel import SumoKernel
from sumospace.settings import SumoSettings
from sumospace.utils.tokens import estimate_tokens

from unittest.mock import patch, AsyncMock

def test_context_truncation_logic(tmp_path):
    settings = SumoSettings(workspace=str(tmp_path))
    with patch("sumospace.kernel.ProviderRouter"), \
         patch("sumospace.rag.RAGPipeline"), \
         patch("sumospace.classifier.SumoClassifier"), \
         patch("sumospace.memory.MemoryManager"), \
         patch("sumospace.ingest.UniversalIngestor"):
        kernel = SumoKernel(settings=settings)
    
    task = "Find the bug"
    # Large contexts to force truncation
    # Budget: RAG 40% (1638 tokens), Web 15% (614 tokens), Memory 20% (819 tokens)
    # Total max_tokens = 4096
    
    # 4 chars per token -> RAG budget ~ 6552 chars
    rag_large = "DATA " * 2000 # 10,000 chars -> should be truncated
    web_large = "WEB " * 1000  # 4,000 chars -> should be truncated
    mem_large = "MEM " * 1000  # 4,000 chars -> should be truncated
    
    full_ctx = kernel._build_full_context(
        task=task,
        rag_context=rag_large,
        web_context=web_large,
        memory_str=mem_large,
        max_tokens=4096
    )
    
    # Verify presence of headers
    assert "=== TASK ===" in full_ctx
    assert "=== RECENT MEMORY ===" in full_ctx
    assert "=== WEB SEARCH RESULTS ===" in full_ctx
    assert "=== CODEBASE CONTEXT ===" in full_ctx
    
    # Verify task is fully present
    assert task in full_ctx
    
    # Verify total size is roughly within limits
    # (4096 tokens * 4 chars/token * 1.2 safety) -> ~19k chars max
    # Actually my estimate_tokens uses 1.2 safety, so truncation is conservative.
    assert estimate_tokens(full_ctx) <= 4500 # A bit over 4096 due to headers/safety
    
    # Verify truncation happened (lengths should be smaller than originals)
    assert len(full_ctx) < len(rag_large) + len(web_large) + len(mem_large)
