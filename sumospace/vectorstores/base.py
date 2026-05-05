# sumospace/vectorstores/base.py
"""
Vector Store Abstraction
========================
All vector store backends implement this interface.
Swap backends via settings.vector_store without changing any retrieval code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorDocument:
    """A document to be stored or retrieved from a vector store."""
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass
class VectorSearchResult:
    """A single search result from a vector store query."""
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity: higher is better


class BaseVectorStore(ABC):
    """
    Abstract interface for all vector store backends.

    Implementations:
        ChromaVectorStore  — default, zero extra deps
        FAISSVectorStore   — pip install sumospace[faiss]
        QdrantVectorStore  — pip install sumospace[qdrant]
        PgVectorStore      — coming in v0.3
    """

    @abstractmethod
    async def add_documents(self, documents: list[VectorDocument]) -> None:
        """Add or upsert documents into the store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[VectorSearchResult]:
        """Semantic search by embedding vector. Returns top_k results."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    async def delete_where(self, filter: dict) -> None:
        """Delete all documents matching a metadata filter."""
        ...

    @abstractmethod
    async def update(self, document: VectorDocument) -> None:
        """Update a single document (upsert by ID)."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove ALL documents from the store. Irreversible."""
        ...

    @abstractmethod
    async def persist(self) -> None:
        """Flush any in-memory state to disk (no-op for persistent stores)."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored documents."""
        ...
