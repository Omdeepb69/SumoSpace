# tests/test_scope.py

"""
Tests for ScopeManager — path resolution, session lifecycle,
TTL cleanup, quota enforcement, and concurrent ingestion locking.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sumospace.scope import ScopeManager, SessionInfo, ScopeStats


# ─── Path Resolution ─────────────────────────────────────────────────────────

class TestPathResolution:
    """Test that resolve() produces correct paths for all three levels."""

    def test_user_level(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="user")
        result = scope.resolve(user_id="alice")
        expected = str(tmp_path / "users" / "alice" / "persistent.db")
        assert result == expected

    def test_session_level(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        result = scope.resolve(user_id="alice", session_id="sess_001")
        expected = str(tmp_path / "users" / "alice" / "sessions" / "sess_001.db")
        assert result == expected

    def test_project_level(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="project")
        result = scope.resolve(user_id="alice", project_id="proj_x")
        expected = str(
            tmp_path / "projects" / "proj_x" / "users" / "alice" / "persistent.db"
        )
        assert result == expected

    def test_user_level_creates_parent_dirs(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="user")
        result = scope.resolve(user_id="bob")
        assert Path(result).parent.exists()

    def test_session_level_creates_parent_dirs(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        result = scope.resolve(user_id="bob", session_id="s1")
        assert Path(result).parent.exists()

    def test_project_level_creates_parent_dirs(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="project")
        result = scope.resolve(user_id="bob", project_id="p1")
        assert Path(result).parent.exists()

    def test_user_level_missing_user_id(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="user")
        with pytest.raises(ValueError, match="user_id required"):
            scope.resolve()

    def test_session_level_missing_session_id(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        with pytest.raises(ValueError, match="session_id required"):
            scope.resolve(user_id="alice")

    def test_session_level_missing_user_id(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        with pytest.raises(ValueError, match="user_id required"):
            scope.resolve(session_id="s1")

    def test_project_level_missing_project_id(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="project")
        with pytest.raises(ValueError, match="project_id required"):
            scope.resolve(user_id="alice")

    def test_project_level_missing_user_id(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="project")
        with pytest.raises(ValueError, match="user_id required"):
            scope.resolve(project_id="p1")

    def test_invalid_level(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid scope level"):
            ScopeManager(chroma_base=str(tmp_path), level="galaxy")

    def test_different_users_get_different_paths(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="user")
        p1 = scope.resolve(user_id="alice")
        p2 = scope.resolve(user_id="bob")
        assert p1 != p2

    def test_different_sessions_get_different_paths(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        p1 = scope.resolve(user_id="alice", session_id="s1")
        p2 = scope.resolve(user_id="alice", session_id="s2")
        assert p1 != p2

    def test_different_projects_get_different_paths(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="project")
        p1 = scope.resolve(user_id="alice", project_id="p1")
        p2 = scope.resolve(user_id="alice", project_id="p2")
        assert p1 != p2


# ─── Session Registry ────────────────────────────────────────────────────────

class TestSessionRegistry:
    """Test that session registration and registry.json work correctly."""

    def test_session_resolve_creates_registry(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        reg_path = tmp_path / "users" / "alice" / "registry.json"
        assert reg_path.exists()
        data = json.loads(reg_path.read_text())
        assert "s1" in data

    def test_multiple_sessions_tracked(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        scope.resolve(user_id="alice", session_id="s2")
        scope.resolve(user_id="alice", session_id="s3")
        data = scope._load_registry("alice")
        assert len(data) == 3

    def test_duplicate_resolve_does_not_duplicate(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        scope.resolve(user_id="alice", session_id="s1")
        data = scope._load_registry("alice")
        assert len(data) == 1

    def test_concurrent_session_registration_no_corruption(self, tmp_path):
        import threading
        import uuid
        
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        errors = []
        
        def register():
            try:
                scope.resolve(user_id="alice", session_id=uuid.uuid4().hex)
            except Exception as e:
                errors.append(e)
                
        threads = [threading.Thread(target=register) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        assert not errors


# ─── Session Listing ─────────────────────────────────────────────────────────

class TestListSessions:
    """Test list_sessions returns correct session info."""

    def test_list_empty(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        sessions = scope.list_sessions("alice")
        assert sessions == []

    def test_list_sessions(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        scope.resolve(user_id="alice", session_id="s2")
        sessions = scope.list_sessions("alice")
        assert len(sessions) == 2
        sids = {s.session_id for s in sessions}
        assert sids == {"s1", "s2"}

    def test_list_sessions_has_timestamps(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        sessions = scope.list_sessions("alice")
        assert sessions[0].created_at
        # Should be valid ISO format
        datetime.fromisoformat(sessions[0].created_at)

    def test_list_sessions_returns_session_info(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        sessions = scope.list_sessions("alice")
        assert isinstance(sessions[0], SessionInfo)
        assert sessions[0].session_id == "s1"


# ─── Hard Delete ──────────────────────────────────────────────────────────────

class TestDeleteSession:
    """Test delete_session performs hard delete."""

    def test_delete_session(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        # Create a dummy file to simulate the db
        db_path = tmp_path / "users" / "alice" / "sessions" / "s1.db"
        db_path.mkdir(parents=True, exist_ok=True)
        (db_path / "data.bin").write_bytes(b"test")

        result = scope.delete_session("alice", "s1")
        assert result is True
        assert not db_path.exists()

    def test_delete_removes_from_registry(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        scope.delete_session("alice", "s1")
        registry = scope._load_registry("alice")
        assert "s1" not in registry

    def test_delete_nonexistent_returns_false(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        result = scope.delete_session("alice", "nonexistent")
        assert result is False


# ─── Archive (Soft Delete) ────────────────────────────────────────────────────

class TestArchiveSession:
    """Test archive_session moves to archive dir."""

    def test_archive_session(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        # Create dummy db content
        db_path = tmp_path / "users" / "alice" / "sessions" / "s1.db"
        db_path.mkdir(parents=True, exist_ok=True)
        (db_path / "data.bin").write_bytes(b"test")

        archive_path = scope.archive_session("alice", "s1")
        assert Path(archive_path).exists()
        assert "archive" in archive_path
        assert not db_path.exists()

    def test_archive_removes_from_registry(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        db_path = tmp_path / "users" / "alice" / "sessions" / "s1.db"
        db_path.mkdir(parents=True, exist_ok=True)

        scope.archive_session("alice", "s1")
        registry = scope._load_registry("alice")
        assert "s1" not in registry

    def test_archive_path_structure(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        db_path = tmp_path / "users" / "alice" / "sessions" / "s1.db"
        db_path.mkdir(parents=True, exist_ok=True)

        archive_path = scope.archive_session("alice", "s1")
        expected = str(tmp_path / "archive" / "alice" / "s1.db")
        assert archive_path == expected


# ─── TTL Cleanup ──────────────────────────────────────────────────────────────

class TestCleanupExpired:
    """Test cleanup_expired removes sessions older than TTL."""

    def test_cleanup_expired_removes_old(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s_old")

        # Backdate the registry entry to 48 hours ago
        registry = scope._load_registry("alice")
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        registry["s_old"] = old_time.isoformat()
        scope._save_registry("alice", registry)

        expired = scope.cleanup_expired("alice", ttl_hours=24)
        assert "s_old" in expired
        assert scope._load_registry("alice").get("s_old") is None

    def test_cleanup_keeps_recent(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s_new")

        expired = scope.cleanup_expired("alice", ttl_hours=24)
        assert expired == []
        assert "s_new" in scope._load_registry("alice")

    def test_cleanup_mixed(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s_old")
        scope.resolve(user_id="alice", session_id="s_new")

        # Backdate only s_old
        registry = scope._load_registry("alice")
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        registry["s_old"] = old_time.isoformat()
        scope._save_registry("alice", registry)

        expired = scope.cleanup_expired("alice", ttl_hours=24)
        assert "s_old" in expired
        assert "s_new" not in expired
        remaining = scope._load_registry("alice")
        assert "s_new" in remaining
        assert "s_old" not in remaining


# ─── Stats ────────────────────────────────────────────────────────────────────

class TestGetStats:
    """Test get_stats returns correct info."""

    def test_stats_empty(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        stats = scope.get_stats("alice")
        assert isinstance(stats, ScopeStats)
        assert stats.total_sessions == 0
        assert stats.total_disk_mb == 0.0

    def test_stats_with_sessions(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        scope.resolve(user_id="alice", session_id="s2")
        stats = scope.get_stats("alice")
        assert stats.total_sessions == 2
        assert set(stats.session_ids) == {"s1", "s2"}
        assert stats.user_id == "alice"

    def test_stats_disk_usage(self, tmp_path):
        scope = ScopeManager(chroma_base=str(tmp_path), level="session")
        scope.resolve(user_id="alice", session_id="s1")
        # Create enough dummy data to register above 0 MB after rounding
        db_path = tmp_path / "users" / "alice" / "sessions" / "s1.db"
        db_path.mkdir(parents=True, exist_ok=True)
        (db_path / "data.bin").write_bytes(b"x" * (1024 * 1024))  # 1 MB

        stats = scope.get_stats("alice")
        assert stats.total_disk_mb >= 1.0


# ─── Quota Exceeded Error ────────────────────────────────────────────────────

class TestQuotaExceeded:
    """Test QuotaExceededError is raised when max_chunks is exceeded."""

    def test_quota_error_attributes(self):
        from sumospace.exceptions import QuotaExceededError
        err = QuotaExceededError(current=100, attempted=50, limit=120)
        assert err.current == 100
        assert err.attempted == 50
        assert err.limit == 120
        assert "100" in str(err)
        assert "50" in str(err)
        assert "120" in str(err)

    def test_quota_error_is_sumo_error(self):
        from sumospace.exceptions import QuotaExceededError, SumoError
        err = QuotaExceededError(current=10, attempted=5, limit=12)
        assert isinstance(err, SumoError)

    @pytest.mark.asyncio
    async def test_ingestor_raises_quota_exceeded(self, tmp_path):
        """Verify UniversalIngestor raises QuotaExceededError at the limit."""
        pytest.importorskip("chromadb")
        from sumospace.exceptions import QuotaExceededError
        from sumospace.ingest import UniversalIngestor, Chunk

        db_path = str(tmp_path / "quota_test_db")
        ingestor = UniversalIngestor(
            chroma_path=db_path,
            collection_name="quota_test",
            max_chunks=5,
        )
        await ingestor.initialize()

        # Create a test file with many lines to generate > 5 chunks
        test_file = tmp_path / "big.txt"
        test_file.write_text("\n\n".join(f"Paragraph {i}. " * 50 for i in range(20)))

        with pytest.raises(QuotaExceededError):
            await ingestor.ingest_file(str(test_file))


# ─── Concurrent Ingestion Lock ───────────────────────────────────────────────

class TestConcurrentLock:
    """Test that asyncio.Lock prevents concurrent upsert races."""

    @pytest.mark.asyncio
    async def test_lock_serializes_ingestion(self, tmp_path):
        """Two concurrent ingestions to the same collection should be serialized."""
        pytest.importorskip("chromadb")
        from sumospace.ingest import UniversalIngestor, _get_collection_lock

        db_path = str(tmp_path / "lock_test_db")
        lock = _get_collection_lock(db_path)

        # Verify the same lock is returned for the same path
        lock2 = _get_collection_lock(db_path)
        assert lock is lock2

        # Different path gets a different lock
        lock3 = _get_collection_lock(str(tmp_path / "other_db"))
        assert lock is not lock3

    @pytest.mark.asyncio
    async def test_concurrent_ingest_does_not_corrupt(self, tmp_path):
        """Two concurrent ingestions should both complete without errors."""
        pytest.importorskip("chromadb")
        from sumospace.ingest import UniversalIngestor

        db_path = str(tmp_path / "concurrent_db")

        # Create two test files
        f1 = tmp_path / "file1.txt"
        f1.write_text("File one content for ingestion test.")
        f2 = tmp_path / "file2.txt"
        f2.write_text("File two content for ingestion test.")

        ingestor = UniversalIngestor(
            chroma_path=db_path,
            collection_name="concurrent_test",
        )
        await ingestor.initialize()

        # Ingest both concurrently
        r1, r2 = await asyncio.gather(
            ingestor.ingest_file(str(f1)),
            ingestor.ingest_file(str(f2)),
        )

        assert r1.chunks_created > 0
        assert r2.chunks_created > 0
        assert not r1.errors
        assert not r2.errors


# ─── Backward Compatibility ──────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Ensure existing code without user_id still works."""

    def test_kernel_config_defaults(self):
        pytest.importorskip("chromadb")
        from sumospace.kernel import KernelConfig
        cfg = KernelConfig()
        assert cfg.user_id == ""
        assert cfg.session_id == ""
        assert cfg.project_id == ""
        assert cfg.scope_level == "user"
        assert cfg.max_chunks_per_scope is None

    def test_memory_manager_no_scope(self):
        """MemoryManager without scope_manager should work as before."""
        pytest.importorskip("chromadb")
        from sumospace.memory import MemoryManager
        mm = MemoryManager(chroma_path="/tmp/test_compat_db")
        assert mm.scope_manager is None
        assert mm.session_id  # Should auto-generate

    def test_ingestor_no_max_chunks(self):
        """UniversalIngestor without max_chunks should default to None."""
        pytest.importorskip("chromadb")
        from sumospace.ingest import UniversalIngestor
        ing = UniversalIngestor(chroma_path="/tmp/test_compat_db2")
        assert ing.max_chunks is None
