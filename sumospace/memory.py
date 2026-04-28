# sumospace/memory.py

"""
Memory System
=============
Two-tier memory:
  - WorkingMemory: In-process ring buffer of recent messages/actions (fast, ephemeral).
  - EpisodicMemory: ChromaDB-backed long-term storage (persistent, semantic search).

Default embeddings: local sentence-transformers (no API key).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    role: str           # "user" | "assistant" | "tool" | "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""

    @property
    def id(self) -> str:
        raw = f"{self.role}:{self.content}:{self.timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }


# ─── Working Memory (in-process ring buffer) ──────────────────────────────────

class WorkingMemory:
    """
    Short-term, in-process ring buffer. Holds the most recent N messages.
    Used to build conversation context for LLM calls.
    """

    def __init__(self, max_size: int = 20):
        self._buffer: deque[MemoryEntry] = deque(maxlen=max_size)

    def add(self, entry: MemoryEntry):
        self._buffer.append(entry)

    def add_message(self, role: str, content: str, **metadata):
        self._buffer.append(MemoryEntry(role=role, content=content, metadata=metadata))

    def recent(self, n: int | None = None) -> list[MemoryEntry]:
        entries = list(self._buffer)
        return entries[-n:] if n else entries

    def as_messages(self, n: int | None = None) -> list[dict[str, str]]:
        """Return as OpenAI/Anthropic-compatible message list."""
        return [
            {"role": e.role, "content": e.content}
            for e in self.recent(n)
        ]

    def clear(self):
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def summary(self) -> str:
        lines = [f"[{e.role}] {e.content[:120]}" for e in self._buffer]
        return "\n".join(lines)


# ─── Episodic Memory (ChromaDB-backed, persistent) ────────────────────────────

class EpisodicMemory:
    """
    Long-term, persistent memory backed by ChromaDB.
    Supports semantic recall across sessions.

    Default embeddings: local sentence-transformers (zero API key).
    """

    def __init__(
        self,
        chroma_path: str = ".sumo_db",
        embedding_provider: str = "local",  # local by default — zero API key
        working_memory_size: int = 20,
    ):
        self.chroma_path = chroma_path
        self.embedding_provider = embedding_provider
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embedder = None
        self.working = WorkingMemory(max_size=working_memory_size)

    async def initialize(self):
        self._client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        await self._init_embedder()

    async def _init_embedder(self):
        if self.embedding_provider in ("local", "sentence-transformers", "st"):
            from sumospace.ingest import LocalEmbeddingProvider
            self._embedder = LocalEmbeddingProvider()
        elif self.embedding_provider == "google":
            from sumospace.ingest import GoogleEmbeddingProvider
            self._embedder = GoogleEmbeddingProvider()
        elif self.embedding_provider == "openai":
            from sumospace.ingest import OpenAIEmbeddingProvider
            self._embedder = OpenAIEmbeddingProvider()
        else:
            from sumospace.ingest import LocalEmbeddingProvider
            self._embedder = LocalEmbeddingProvider()

    async def store(self, entry: MemoryEntry):
        """Persist an entry to long-term memory and add to working memory."""
        self.working.add(entry)
        embeddings = await self._embedder.embed([entry.content])
        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.content],
            embeddings=embeddings,
            metadatas=[{
                "role": entry.role,
                "timestamp": entry.timestamp,
                "session_id": entry.session_id,
                **{k: str(v) for k, v in entry.metadata.items()},
            }],
        )

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Semantic recall from long-term memory."""
        embeddings = await self._embedder.embed([query])
        query_kwargs: dict = {
            "query_embeddings": embeddings,
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if session_id:
            query_kwargs["where"] = {"session_id": session_id}

        results = self._collection.query(**query_kwargs)
        entries = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            entry = MemoryEntry(
                role=meta.get("role", "unknown"),
                content=doc,
                metadata={"similarity": 1 - dist, **meta},
                timestamp=float(meta.get("timestamp", 0)),
                session_id=meta.get("session_id", ""),
            )
            entries.append(entry)
        return entries

    async def store_message(self, role: str, content: str, session_id: str = "", **meta):
        entry = MemoryEntry(role=role, content=content,
                            metadata=meta, session_id=session_id)
        await self.store(entry)

    def recent_context(self, n: int = 10) -> str:
        """Return recent working memory as a formatted context string."""
        return self.working.summary()

    @property
    def count(self) -> int:
        if self._collection:
            return self._collection.count()
        return 0


# ─── Unified Memory Manager ───────────────────────────────────────────────────

class MemoryManager:
    """
    Top-level manager that combines working + episodic memory.
    Used by the Kernel to track conversation history and recall relevant context.

    When a ScopeManager is provided, all chroma_path resolution is delegated to it
    instead of using the raw chroma_path string. This enables multi-user/session/project
    isolation at the filesystem level.
    """

    def __init__(
        self,
        chroma_path: str = ".sumo_db",
        embedding_provider: str = "local",
        working_size: int = 20,
        session_id: str = "",
        scope_manager: "ScopeManager | None" = None,
        user_id: str = "",
        project_id: str = "",
    ):
        self.session_id = session_id or _generate_session_id()
        self.scope_manager = scope_manager

        # If a ScopeManager is provided, resolve chroma_path through it
        if scope_manager is not None:
            chroma_path = scope_manager.resolve(
                user_id=user_id,
                session_id=self.session_id,
                project_id=project_id,
            )

        self.episodic = EpisodicMemory(
            chroma_path=chroma_path,
            embedding_provider=embedding_provider,
            working_memory_size=working_size,
        )

    async def initialize(self):
        await self.episodic.initialize()

    async def add(self, role: str, content: str, **meta):
        await self.episodic.store_message(
            role=role, content=content,
            session_id=self.session_id, **meta,
        )

    async def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return await self.episodic.recall(query, top_k=top_k)

    def recent(self, n: int = 10) -> list[dict[str, str]]:
        return self.episodic.working.as_messages(n)

    def context_string(self, n: int = 10) -> str:
        return self.episodic.recent_context(n)

    @property
    def working(self) -> WorkingMemory:
        return self.episodic.working


def _generate_session_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
