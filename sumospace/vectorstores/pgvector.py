# sumospace/vectorstores/pgvector.py
"""
pgvector Vector Store — STUB
=============================
pgvector support is planned for SumoSpace v0.3.

The settings field `vector_store = "pgvector"` and `vector_store_url` are
already available so you can configure it today without breaking changes.

v0.3 will implement:
    - Full PostgreSQL + pgvector extension support
    - Connection pooling via asyncpg
    - Hybrid BM25 + dense retrieval
    - Persistent metadata in the same Postgres database as your app data

For now, attempting to use this backend will raise NotImplementedError
with a clear migration message.
"""
from __future__ import annotations

from sumospace.vectorstores.base import BaseVectorStore, VectorDocument, VectorSearchResult

_NOT_IMPLEMENTED_MSG = (
    "pgvector support is coming in SumoSpace v0.3. "
    "In the meantime, use 'chroma' (default) or 'qdrant' for production workloads. "
    "See: https://github.com/Omdeepb69/SumoSpace/issues for updates."
)


class PgVectorStore(BaseVectorStore):
    """pgvector backend — planned for v0.3."""

    def __init__(self, settings):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def add_documents(self, documents: list[VectorDocument]) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def search(self, query_embedding, top_k=10, where=None):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def delete(self, ids):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def delete_where(self, filter):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def update(self, document):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def clear(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def persist(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def count(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
