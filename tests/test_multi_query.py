# tests/test_multi_query.py
"""Tests for multi-query retrieval expansion."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sumospace.rag import RAGPipeline
from sumospace.settings import SumoSettings


@pytest.fixture
def mock_ingestor():
    ingestor = MagicMock()
    ingestor.query = AsyncMock(return_value=[
        {"text": "chunk one", "metadata": {"source": "a.py"}, "score": 0.9},
        {"text": "chunk two", "metadata": {"source": "b.py"}, "score": 0.8},
    ])
    return ingestor


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=(
        "How does auth work?\nWhat is the login mechanism?\nDescribe authentication"
    ))
    return provider


@pytest.mark.asyncio
async def test_single_query_mode(mock_ingestor):
    """Default mode: single query, no expansion."""
    settings = SumoSettings(rag_multi_query=False)
    pipeline = RAGPipeline(
        ingestor=mock_ingestor,
        use_reranker=False,
        settings=settings,
    )
    result = await pipeline.retrieve("how does auth work?")
    assert mock_ingestor.query.call_count == 1
    assert len(result.chunks) == 2


@pytest.mark.asyncio
async def test_multi_query_mode_calls_provider(mock_ingestor, mock_provider):
    """Multi-query mode calls provider to expand query."""
    settings = SumoSettings(rag_multi_query=True, rag_multi_query_count=3)
    pipeline = RAGPipeline(
        ingestor=mock_ingestor,
        use_reranker=False,
        settings=settings,
        provider=mock_provider,
    )
    result = await pipeline.retrieve("how does auth work?")
    # Provider called once for expansion
    mock_provider.complete.assert_called_once()
    # Ingestor called at least twice (original + variants)
    assert mock_ingestor.query.call_count >= 2


@pytest.mark.asyncio
async def test_multi_query_deduplication(mock_ingestor, mock_provider):
    """Duplicate chunks from different query variants should be deduplicated."""
    settings = SumoSettings(rag_multi_query=True, rag_multi_query_count=2)
    # Return identical results for every query
    pipeline = RAGPipeline(
        ingestor=mock_ingestor,
        use_reranker=False,
        settings=settings,
        provider=mock_provider,
    )
    result = await pipeline.retrieve("auth flow")
    # Despite multiple queries returning same chunks, dedup keeps only unique
    texts = [c.text for c in result.chunks]
    assert len(texts) == len(set(texts))


@pytest.mark.asyncio
async def test_multi_query_provider_failure_falls_back(mock_ingestor):
    """If provider fails, falls back to single-query mode gracefully."""
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    settings = SumoSettings(rag_multi_query=True, rag_multi_query_count=3)
    pipeline = RAGPipeline(
        ingestor=mock_ingestor,
        use_reranker=False,
        settings=settings,
        provider=provider,
    )
    # Should not raise — falls back to single query
    result = await pipeline.retrieve("query")
    assert len(result.chunks) > 0
    assert mock_ingestor.query.call_count == 1


@pytest.mark.asyncio
async def test_multi_query_disabled_without_provider(mock_ingestor):
    """Multi-query is skipped if provider is None even when setting is True."""
    settings = SumoSettings(rag_multi_query=True, rag_multi_query_count=3)
    pipeline = RAGPipeline(
        ingestor=mock_ingestor,
        use_reranker=False,
        settings=settings,
        provider=None,  # No provider
    )
    result = await pipeline.retrieve("query")
    # Falls back to single query
    assert mock_ingestor.query.call_count == 1
