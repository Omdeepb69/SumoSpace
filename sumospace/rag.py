# sumospace/rag.py

"""
Retrieval-Augmented Generation (RAG) Pipeline
===============================================
Two-stage retrieval:
  Stage 1 — Vector search (ChromaDB, cosine similarity, top-K candidates)
  Stage 2 — Cross-encoder reranking (local, no API key)

Assembles a final context block for LLM consumption.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from sumospace.ingest import UniversalIngestor

if TYPE_CHECKING:
    from sumospace.settings import SumoSettings


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    vector_score: float
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.vector_score

    def to_context_block(self) -> str:
        source = self.metadata.get("source", "unknown")
        page = self.metadata.get("page", "")
        page_str = f" (page {page})" if page else ""
        return f"[Source: {source}{page_str} | Score: {self.final_score:.3f}]\n{self.text}"


@dataclass
class RAGResult:
    query: str
    chunks: list[RetrievedChunk]
    context: str
    used_reranker: bool = False
    total_candidates: int = 0


# ─── Cross-Encoder Reranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence-transformers.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (~80MB, CPU-friendly)

    No API key. No internet at inference.
    Much more accurate than cosine similarity alone for ranking.
    """
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or self.MODEL
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Rerank chunks by relevance to query. Returns sorted descending."""
        if not chunks:
            return chunks

        self._ensure_model()
        pairs = [(query, c.text) for c in chunks]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs),
        )

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        return sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline: vector search → rerank → context assembly.

    Usage:
        pipeline = RAGPipeline(ingestor)
        await pipeline.initialize()
        result = await pipeline.retrieve("How does the auth module work?")
        print(result.context)
    """

    def __init__(
        self,
        ingestor: UniversalIngestor,
        top_k_candidates: int = 20,
        top_k_final: int = 5,
        reranker_model: str | None = None,
        use_reranker: bool = True,
        max_context_chars: int = 6000,
        settings: "SumoSettings | None" = None,
        provider=None,
    ):
        self.ingestor = ingestor
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final
        self.use_reranker = use_reranker
        self.max_context_chars = max_context_chars
        self._settings = settings
        self._provider = provider
        self._reranker: CrossEncoderReranker | None = None

    async def initialize(self):
        if self.use_reranker:
            self._reranker = CrossEncoderReranker()

    async def retrieve(
        self,
        query: str,
        filter_metadata: dict | None = None,
        force_no_rerank: bool = False,
    ) -> RAGResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query:           Natural language query.
            filter_metadata: Optional ChromaDB where-filter (e.g., {"type": "function"}).
            force_no_rerank: Skip reranker even if configured.
        """
        # Multi-query expansion (optional)
        queries = [query]
        if (
            self._settings is not None
            and self._settings.rag_multi_query
            and self._provider is not None
        ):
            variants = await self._expand_query(
                query, self._settings.rag_multi_query_count
            )
            queries = [query] + variants

        # Stage 1 — Vector search (all query variants)
        all_raw: list[dict] = []
        for q in queries:
            raw = await self.ingestor.query(
                query_text=q,
                top_k=self.top_k_candidates,
                where=filter_metadata,
            )
            all_raw.extend(raw)

        # Deduplicate by keeping best score per unique text
        candidates = self._deduplicate(all_raw)

        # Stage 2 — Cross-encoder rerank
        used_reranker = False
        if self._reranker and not force_no_rerank and len(candidates) > 1:
            candidates = await self._reranker.rerank(query, candidates)
            used_reranker = True

        final_chunks = candidates[:self.top_k_final]

        # Context assembly — trim to max_context_chars
        context_parts = []
        total_chars = 0
        for chunk in final_chunks:
            block = chunk.to_context_block()
            if total_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining > 100:
                    context_parts.append(block[:remaining] + "...[truncated]")
                break
            context_parts.append(block)
            total_chars += len(block)

        context = "\n\n---\n\n".join(context_parts)

        return RAGResult(
            query=query,
            chunks=final_chunks,
            context=context,
            used_reranker=used_reranker,
            total_candidates=len(all_raw),
        )

    # ── Multi-query helpers ───────────────────────────────────────────────────

    async def _expand_query(self, query: str, n: int = 3) -> list[str]:
        """Generate n alternative phrasings of query using the current LLM provider."""
        try:
            prompt = (
                f"Generate {n} different phrasings of this search query. "
                f"Output only the queries, one per line, no numbering, no explanation.\n\n"
                f"Query: {query}"
            )
            response = await self._provider.complete(
                user=prompt,
                temperature=0.7,
                max_tokens=150,
            )
            variants = [
                line.strip() for line in response.splitlines()
                if line.strip() and line.strip() != query
            ]
            return variants[:n]
        except Exception:
            return []  # fallback: single-query mode

    def _deduplicate(self, raw_results: list[dict]) -> list[RetrievedChunk]:
        """Merge multi-query results, keeping the highest score per unique text."""
        seen: dict[str, RetrievedChunk] = {}
        for r in raw_results:
            key = r["text"][:100]  # first 100 chars as dedup key
            chunk = RetrievedChunk(
                text=r["text"],
                metadata=r["metadata"],
                vector_score=r["score"],
            )
            if key not in seen or chunk.vector_score > seen[key].vector_score:
                seen[key] = chunk
        return sorted(seen.values(), key=lambda c: c.vector_score, reverse=True)

    def build_prompt(
        self,
        query: str,
        rag_result: RAGResult,
        task_description: str = "",
    ) -> str:
        """
        Assemble a full RAG prompt combining retrieved context + user query.
        """
        header = task_description or "Answer the user's question using the retrieved context below."
        return (
            f"{header}\n\n"
            f"=== RETRIEVED CONTEXT ===\n{rag_result.context}\n"
            f"=========================\n\n"
            f"USER QUERY: {query}"
        )
