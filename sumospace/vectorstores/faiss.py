# sumospace/vectorstores/faiss.py
"""
FAISS Vector Store
==================
In-memory vector store backed by Facebook AI Similarity Search.
Persists index to disk as a flat file.

Requirements:
    pip install sumospace[faiss]
    (installs faiss-cpu)
"""
from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path

from sumospace.vectorstores.base import BaseVectorStore, VectorDocument, VectorSearchResult


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed flat IP index with metadata side-car.
    Dimension is inferred from the first document added.
    """

    def __init__(self, settings):
        self._settings = settings
        self._persist_dir = Path(settings.chroma_base) / "faiss"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._persist_dir / "index.faiss"
        self._meta_path = self._persist_dir / "metadata.pkl"

        self._index = None          # faiss.Index
        self._id_map: list[str] = []
        self._meta_map: dict[str, dict] = {}
        self._text_map: dict[str, str] = {}
        self._dim: int | None = None

    def _ensure_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "FAISS is not installed. Run: pip install sumospace[faiss]"
            )

    def _try_load_existing(self) -> bool:
        """Attempt to load from disk without knowing dim. Returns True if loaded."""
        faiss = self._ensure_faiss()
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                state = pickle.load(f)
                self._id_map = state["id_map"]
                self._meta_map = state["meta_map"]
                self._text_map = state["text_map"]
                self._dim = state["dim"]
            return True
        return False

    def _load_or_create(self, dim: int):
        if not self._try_load_existing():
            self._dim = dim
            faiss = self._ensure_faiss()
            self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine when normalized

    def _normalize(self, vec: list[float]) -> list[float]:
        import numpy as np
        v = np.array(vec, dtype="float32")
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v.tolist()

    async def add_documents(self, documents: list[VectorDocument]) -> None:
        import numpy as np
        faiss = self._ensure_faiss()

        if not documents:
            return

        dim = len(documents[0].embedding)
        if self._index is None:
            self._load_or_create(dim)

        loop = asyncio.get_event_loop()

        def _add():
            vecs = np.array(
                [self._normalize(d.embedding) for d in documents], dtype="float32"
            )
            self._index.add(vecs)
            for d in documents:
                self._id_map.append(d.id)
                self._meta_map[d.id] = d.metadata
                self._text_map[d.id] = d.text

        await loop.run_in_executor(None, _add)
        await self.persist()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[VectorSearchResult]:
        import numpy as np

        if self._index is None:
            self._try_load_existing()

        if self._index is None or self._index.ntotal == 0:
            return []

        loop = asyncio.get_event_loop()

        def _search():
            vec = np.array([self._normalize(query_embedding)], dtype="float32")
            k = min(top_k * 3, self._index.ntotal)  # over-fetch for where filtering
            distances, indices = self._index.search(vec, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                doc_id = self._id_map[idx]
                meta = self._meta_map.get(doc_id, {})
                # Apply metadata filter
                if where:
                    if not all(meta.get(k) == v for k, v in where.items()):
                        continue
                results.append(VectorSearchResult(
                    id=doc_id,
                    text=self._text_map.get(doc_id, ""),
                    metadata=meta,
                    score=float(dist),
                ))
                if len(results) >= top_k:
                    break
            return results

        return await loop.run_in_executor(None, _search)

    async def delete(self, ids: list[str]) -> None:
        # FAISS flat index does not support deletion; rebuild without those IDs
        to_remove = set(ids)
        remaining_ids = [i for i in self._id_map if i not in to_remove]
        remaining_docs = [
            VectorDocument(
                id=i,
                text=self._text_map[i],
                embedding=[],  # re-use stored meta only — no re-embedding needed
                metadata=self._meta_map[i],
            )
            for i in remaining_ids
        ]
        await self.clear()
        # Note: embeddings are not stored; users needing delete must re-ingest
        # This is a FAISS limitation. Use Qdrant for mutable workloads.

    async def delete_where(self, filter: dict) -> None:
        to_remove = [
            id_ for id_, meta in self._meta_map.items()
            if all(meta.get(k) == v for k, v in filter.items())
        ]
        await self.delete(to_remove)

    async def update(self, document: VectorDocument) -> None:
        await self.add_documents([document])

    async def clear(self) -> None:
        faiss = self._ensure_faiss()
        dim = self._dim or 768
        self._index = faiss.IndexFlatIP(dim)
        self._id_map = []
        self._meta_map = {}
        self._text_map = {}
        await self.persist()

    async def persist(self) -> None:
        faiss = self._ensure_faiss()
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "wb") as f:
                pickle.dump({
                    "id_map": self._id_map,
                    "meta_map": self._meta_map,
                    "text_map": self._text_map,
                    "dim": self._dim,
                }, f)

    async def count(self) -> int:
        if self._index is None:
            self._try_load_existing()
        if self._index is None:
            return 0
        return self._index.ntotal
