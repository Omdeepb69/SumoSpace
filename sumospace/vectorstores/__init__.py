# sumospace/vectorstores/__init__.py
"""
Pluggable vector store backends for SumoSpace.

Usage:
    from sumospace.vectorstores import get_vector_store
    store = get_vector_store(settings)
"""
from __future__ import annotations

from sumospace.vectorstores.base import BaseVectorStore, VectorDocument, VectorSearchResult


def get_vector_store(settings) -> BaseVectorStore:
    """
    Factory: return the correct vector store based on settings.vector_store.

    Args:
        settings: SumoSettings instance.

    Returns:
        A configured BaseVectorStore implementation.
    """
    backend = settings.vector_store

    if backend == "chroma":
        from sumospace.vectorstores.chroma import ChromaVectorStore
        return ChromaVectorStore(settings)
    elif backend == "faiss":
        from sumospace.vectorstores.faiss import FAISSVectorStore
        return FAISSVectorStore(settings)
    elif backend == "qdrant":
        from sumospace.vectorstores.qdrant import QdrantVectorStore
        return QdrantVectorStore(settings)
    elif backend == "pgvector":
        from sumospace.vectorstores.pgvector import PgVectorStore
        return PgVectorStore(settings)
    else:
        raise ValueError(
            f"Unknown vector store backend: '{backend}'. "
            f"Valid options: 'chroma', 'faiss', 'qdrant', 'pgvector'."
        )


__all__ = [
    "BaseVectorStore",
    "VectorDocument",
    "VectorSearchResult",
    "get_vector_store",
]
