# tests/test_vectorstores.py
"""Tests for vector store abstraction layer."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sumospace.vectorstores import get_vector_store
from sumospace.vectorstores.base import VectorDocument, VectorSearchResult
from sumospace.settings import SumoSettings


@pytest.fixture
def settings(tmp_path):
    return SumoSettings(chroma_base=str(tmp_path / ".sumo_db"), vector_store="chroma")


def test_get_vector_store_chroma(settings):
    from sumospace.vectorstores.chroma import ChromaVectorStore
    store = get_vector_store(settings)
    assert isinstance(store, ChromaVectorStore)


def test_get_vector_store_unknown_raises():
    settings = SumoSettings()
    settings.__dict__["vector_store"] = "unknown_backend"
    with pytest.raises(ValueError, match="Unknown vector store"):
        get_vector_store(settings)


def test_pgvector_raises_not_implemented(settings):
    settings.__dict__["vector_store"] = "pgvector"
    with pytest.raises(NotImplementedError, match="v0.3"):
        get_vector_store(settings)


@pytest.mark.asyncio
async def test_chroma_store_add_and_search(tmp_path):
    settings = SumoSettings(chroma_base=str(tmp_path / ".sumo_db"), vector_store="chroma")
    store = get_vector_store(settings)

    docs = [
        VectorDocument(
            id="doc1",
            text="The quick brown fox",
            embedding=[0.1] * 768,
            metadata={"source": "test.txt"},
        )
    ]

    with patch.object(store, "_ensure_client"):
        store._collection = MagicMock()
        store._collection.count.return_value = 1
        store._collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["The quick brown fox"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.1]],
        }
        results = await store.search([0.1] * 768, top_k=1)

    assert len(results) == 1
    assert results[0].id == "doc1"
    assert results[0].score == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_chroma_empty_db_returns_empty(tmp_path):
    settings = SumoSettings(chroma_base=str(tmp_path / ".sumo_db"), vector_store="chroma")
    store = get_vector_store(settings)

    with patch.object(store, "_ensure_client"):
        store._collection = MagicMock()
        store._collection.count.return_value = 0
        results = await store.search([0.1] * 768, top_k=5)

    assert results == []


@pytest.mark.asyncio
async def test_faiss_store_add_and_search(tmp_path):
    pytest.importorskip("faiss", reason="faiss-cpu not installed")
    from sumospace.vectorstores.faiss import FAISSVectorStore
    settings = SumoSettings(chroma_base=str(tmp_path / ".sumo_db"))
    store = FAISSVectorStore(settings)

    docs = [
        VectorDocument(id="a", text="hello", embedding=[1.0, 0.0], metadata={}),
        VectorDocument(id="b", text="world", embedding=[0.0, 1.0], metadata={}),
    ]
    await store.add_documents(docs)
    results = await store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "a"


@pytest.mark.asyncio
async def test_qdrant_store_import_error(tmp_path):
    """QdrantVectorStore raises ImportError when qdrant_client is absent."""
    settings = SumoSettings(chroma_base=str(tmp_path / ".sumo_db"), vector_store="qdrant")
    from sumospace.vectorstores.qdrant import QdrantVectorStore
    store = QdrantVectorStore(settings)
    with pytest.raises(ImportError, match="qdrant"):
        with patch.dict("sys.modules", {"qdrant_client": None}):
            store._client = None  # reset so _ensure_client re-runs
            store._ensure_client()
