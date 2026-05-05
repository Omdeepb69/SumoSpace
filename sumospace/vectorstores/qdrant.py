# sumospace/vectorstores/qdrant.py
"""
Qdrant Vector Store
===================
Connects to a local or remote Qdrant instance.

Requirements:
    pip install sumospace[qdrant]
    Docker: docker run -p 6333:6333 qdrant/qdrant

Settings:
    vector_store = "qdrant"
    vector_store_url = "http://localhost:6333"   # or your Qdrant Cloud URL
"""
from __future__ import annotations

import asyncio
import uuid

from sumospace.vectorstores.base import BaseVectorStore, VectorDocument, VectorSearchResult

COLLECTION_NAME = "sumospace"


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant-backed vector store with cosine similarity.
    Collection is created on first write if it doesn't exist.
    """

    def __init__(self, settings):
        self._settings = settings
        self._url = settings.vector_store_url or "http://localhost:6333"
        self._client = None
        self._dim: int | None = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "Qdrant client is not installed. Run: pip install sumospace[qdrant]"
                )
            self._client = QdrantClient(url=self._url)

    async def _ensure_collection(self, dim: int):
        from qdrant_client.models import Distance, VectorParams
        loop = asyncio.get_event_loop()

        def _create():
            collections = [c.name for c in self._client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        await loop.run_in_executor(None, _create)

    async def add_documents(self, documents: list[VectorDocument]) -> None:
        from qdrant_client.models import PointStruct
        self._ensure_client()

        if not documents:
            return

        dim = len(documents[0].embedding)
        await self._ensure_collection(dim)

        loop = asyncio.get_event_loop()

        def _upsert():
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, d.id)),
                    vector=d.embedding,
                    payload={"text": d.text, "doc_id": d.id, **d.metadata},
                )
                for d in documents
            ]
            self._client.upsert(collection_name=COLLECTION_NAME, points=points)

        await loop.run_in_executor(None, _upsert)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[VectorSearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self._ensure_client()

        loop = asyncio.get_event_loop()

        def _search():
            qdrant_filter = None
            if where:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in where.items()
                    ]
                )
            return self._client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        try:
            hits = await loop.run_in_executor(None, _search)
        except Exception:
            return []

        return [
            VectorSearchResult(
                id=hit.payload.get("doc_id", str(hit.id)),
                text=hit.payload.get("text", ""),
                metadata={k: v for k, v in hit.payload.items() if k not in ("text", "doc_id")},
                score=hit.score,
            )
            for hit in hits
        ]

    async def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList
        self._ensure_client()
        loop = asyncio.get_event_loop()
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)) for id_ in ids]
        await loop.run_in_executor(
            None,
            lambda: self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=point_ids),
            )
        )

    async def delete_where(self, filter: dict) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
        self._ensure_client()
        loop = asyncio.get_event_loop()
        qdrant_filter = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()]
        )
        await loop.run_in_executor(
            None,
            lambda: self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(filter=qdrant_filter),
            )
        )

    async def update(self, document: VectorDocument) -> None:
        await self.add_documents([document])

    async def clear(self) -> None:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.delete_collection(COLLECTION_NAME)
        )

    async def persist(self) -> None:
        pass  # Qdrant persists automatically

    async def count(self) -> int:
        self._ensure_client()
        loop = asyncio.get_event_loop()
        try:
            info = await loop.run_in_executor(
                None, lambda: self._client.get_collection(COLLECTION_NAME)
            )
            return info.points_count or 0
        except Exception:
            return 0
