# sumospace/vectorstores/chroma.py
"""
ChromaDB Vector Store
=====================
Default backend. Wraps the existing ChromaDB client from ingest.py.
Zero behavior change — existing Chroma collections are reused.
"""
from __future__ import annotations

import asyncio
from typing import Any

from sumospace.vectorstores.base import BaseVectorStore, VectorDocument, VectorSearchResult


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store.

    Uses the same PersistentClient and collection as UniversalIngestor.
    Existing ingested data is immediately accessible without re-ingestion.
    """

    def __init__(self, settings):
        self._settings = settings
        self._client = None
        self._collection = None

    def _ensure_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self._settings.chroma_base,
                settings=Settings(anonymized_telemetry=False),
            )
            collection_name = getattr(self._settings, "chroma_collection", "sumospace")
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    async def add_documents(self, documents: list[VectorDocument]) -> None:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.upsert(
                ids=[d.id for d in documents],
                documents=[d.text for d in documents],
                embeddings=[d.embedding for d in documents],
                metadatas=[d.metadata for d in documents],
            )
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_client()
        loop = asyncio.get_event_loop()

        n = min(top_k, await self.count())
        if n == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = await loop.run_in_executor(
            None, lambda: self._collection.query(**kwargs)
        )

        return [
            VectorSearchResult(
                id=id_,
                text=doc,
                metadata=meta,
                score=1.0 - dist,  # cosine distance → similarity
            )
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    async def delete(self, ids: list[str]) -> None:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.delete(ids=ids))

    async def delete_where(self, filter: dict) -> None:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.delete(where=filter))

    async def update(self, document: VectorDocument) -> None:
        await self.add_documents([document])

    async def clear(self) -> None:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        name = self._collection.name
        await loop.run_in_executor(None, lambda: self._client.delete_collection(name))
        self._collection = None
        self._ensure_client()

    async def persist(self) -> None:
        pass  # ChromaDB PersistentClient auto-flushes

    async def count(self) -> int:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._collection.count)
