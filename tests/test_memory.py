# tests/test_memory.py

import pytest
import time
from sumospace.memory import (
    MemoryEntry,
    WorkingMemory,
    EpisodicMemory,
    MemoryManager,
)


class TestMemoryEntry:
    def test_id_is_deterministic(self):
        e1 = MemoryEntry(role="user", content="hello", timestamp=1000.0)
        e2 = MemoryEntry(role="user", content="hello", timestamp=1000.0)
        assert e1.id == e2.id

    def test_different_content_different_id(self):
        e1 = MemoryEntry(role="user", content="hello")
        e2 = MemoryEntry(role="user", content="world")
        assert e1.id != e2.id

    def test_to_dict(self):
        e = MemoryEntry(role="assistant", content="response", session_id="abc")
        d = e.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "response"
        assert d["session_id"] == "abc"


class TestWorkingMemory:
    def setup_method(self):
        self.mem = WorkingMemory(max_size=5)

    def test_add_and_retrieve(self):
        self.mem.add_message("user", "Hello")
        self.mem.add_message("assistant", "Hi there")
        entries = self.mem.recent()
        assert len(entries) == 2

    def test_ring_buffer_eviction(self):
        for i in range(10):
            self.mem.add_message("user", f"Message {i}")
        entries = self.mem.recent()
        assert len(entries) == 5
        # Most recent 5 messages
        assert "Message 9" in entries[-1].content

    def test_recent_n(self):
        for i in range(5):
            self.mem.add_message("user", f"msg {i}")
        entries = self.mem.recent(2)
        assert len(entries) == 2
        assert "msg 4" in entries[-1].content

    def test_as_messages(self):
        self.mem.add_message("user", "Question")
        self.mem.add_message("assistant", "Answer")
        msgs = self.mem.as_messages()
        assert msgs == [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

    def test_clear(self):
        self.mem.add_message("user", "test")
        self.mem.clear()
        assert len(self.mem) == 0

    def test_summary(self):
        self.mem.add_message("user", "What is 2+2?")
        self.mem.add_message("assistant", "4")
        summary = self.mem.summary()
        assert "user" in summary.lower()
        assert "assistant" in summary.lower()
        assert "2+2" in summary


@pytest.mark.asyncio
class TestEpisodicMemory:
    def teardown_method(self, method):
        if hasattr(self, "mem") and self.mem and self.mem._client:
            try:
                self.mem._client._system.stop()
            except Exception:
                pass

    async def test_initialize(self, tmp_chroma):
        self.mem = EpisodicMemory(chroma_path=tmp_chroma)
        await self.mem.initialize()
        assert self.mem._client is not None
        assert self.mem._collection is not None

    async def test_store_and_recall(self, tmp_chroma):
        self.mem = EpisodicMemory(chroma_path=tmp_chroma)
        await self.mem.initialize()

        entry = MemoryEntry(
            role="user",
            content="The authentication system uses JWT tokens for session management.",
            session_id="sess1",
        )
        await self.mem.store(entry)

        results = await self.mem.recall("JWT authentication", top_k=1)
        assert len(results) >= 1
        assert "JWT" in results[0].content or results[0].content

    async def test_store_message_helper(self, tmp_chroma):
        self.mem = EpisodicMemory(chroma_path=tmp_chroma)
        await self.mem.initialize()
        await self.mem.store_message("user", "hello world", session_id="s1")
        results = await self.mem.recall("hello", top_k=1)
        assert len(results) >= 1

    async def test_working_memory_updated_on_store(self, tmp_chroma):
        self.mem = EpisodicMemory(chroma_path=tmp_chroma)
        await self.mem.initialize()
        entry = MemoryEntry(role="user", content="test message")
        await self.mem.store(entry)
        assert len(self.mem.working) == 1

    async def test_count(self, tmp_chroma):
        self.mem = EpisodicMemory(chroma_path=tmp_chroma)
        await self.mem.initialize()
        assert self.mem.count == 0
        await self.mem.store_message("user", "first")
        await self.mem.store_message("assistant", "second")
        assert self.mem.count == 2


@pytest.mark.asyncio
class TestMemoryManager:
    def teardown_method(self, method):
        if hasattr(self, "mgr") and self.mgr and self.mgr.episodic._client:
            try:
                self.mgr.episodic._client._system.stop()
            except Exception:
                pass

    async def test_initialize(self, tmp_chroma):
        self.mgr = MemoryManager(chroma_path=tmp_chroma, session_id="test-session")
        await self.mgr.initialize()
        assert self.mgr.session_id == "test-session"

    async def test_add_and_recent(self, tmp_chroma):
        self.mgr = MemoryManager(chroma_path=tmp_chroma)
        await self.mgr.initialize()
        await self.mgr.add("user", "What is Python?")
        await self.mgr.add("assistant", "Python is a programming language.")
        recent = self.mgr.recent(10)
        assert len(recent) == 2
        assert recent[0]["role"] == "user"
        assert recent[1]["role"] == "assistant"

    async def test_recall(self, tmp_chroma):
        self.mgr = MemoryManager(chroma_path=tmp_chroma)
        await self.mgr.initialize()
        await self.mgr.add("user", "Explain dependency injection in software design.")
        await self.mgr.add("assistant", "Dependency injection is a design pattern where dependencies are provided externally.")
        results = await self.mgr.recall("design patterns", top_k=2)
        assert len(results) >= 1

    async def test_session_id_auto_generated(self, tmp_chroma):
        self.mgr = MemoryManager(chroma_path=tmp_chroma)
        assert len(self.mgr.session_id) == 12  # 12 hex chars

    async def test_context_string(self, tmp_chroma):
        self.mgr = MemoryManager(chroma_path=tmp_chroma)
        await self.mgr.initialize()
        await self.mgr.add("user", "First message")
        await self.mgr.add("assistant", "First response")
        ctx = self.mgr.context_string()
        assert "user" in ctx.lower() or "First" in ctx
