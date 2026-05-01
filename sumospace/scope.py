# sumospace/scope.py

"""
Scope Manager
=============
Single source of truth for all ChromaDB path and collection name resolution.
Supports three isolation levels:
  - user:    .sumo_db/users/{user_id}/persistent.db
  - session: .sumo_db/users/{user_id}/sessions/{session_id}.db
  - project: .sumo_db/projects/{project_id}/users/{user_id}/persistent.db

Session lifecycle:
  - list_sessions()    → all session db files with created-at timestamp
  - delete_session()   → hard delete
  - archive_session()  → move to .sumo_db/archive/ (soft delete, recoverable)
  - cleanup_expired()  → auto-delete sessions older than TTL

Tracks sessions via a lightweight registry.json in each user directory.
No external database required — stdlib json only.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    """Metadata for a tracked session."""
    session_id: str
    created_at: str          # ISO-8601
    db_path: str
    size_bytes: int = 0

    @property
    def created_timestamp(self) -> float:
        return datetime.fromisoformat(self.created_at).timestamp()


@dataclass
class ScopeStats:
    """Disk usage and collection stats for a user scope."""
    user_id: str
    total_sessions: int
    total_disk_mb: float
    session_ids: list[str] = field(default_factory=list)


# ─── Scope Manager ───────────────────────────────────────────────────────────

VALID_LEVELS = {"user", "session", "project"}


class ScopeManager:
    """
    Resolves ChromaDB paths based on isolation level.

    Usage:
        scope = ScopeManager(chroma_base=".sumo_db", level="user")
        path = scope.resolve(user_id="alice")
        # → .sumo_db/users/alice/persistent.db

        scope = ScopeManager(chroma_base=".sumo_db", level="session")
        path = scope.resolve(user_id="alice", session_id="abc123")
        # → .sumo_db/users/alice/sessions/abc123.db

        scope = ScopeManager(chroma_base=".sumo_db", level="project")
        path = scope.resolve(user_id="alice", project_id="proj1")
        # → .sumo_db/projects/proj1/users/alice/persistent.db
    """

    def __init__(
        self,
        chroma_base: str = ".sumo_db",
        level: str = "user",
    ):
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Invalid scope level {level!r}. Must be one of: {VALID_LEVELS}"
            )
        self.chroma_base = Path(chroma_base)
        self.level = level

    # ── Path Resolution ──────────────────────────────────────────────────────

    def resolve(
        self,
        user_id: str = "",
        session_id: str = "",
        project_id: str = "",
    ) -> str:
        """
        Resolve the ChromaDB persistent path for the given scope parameters.

        Returns an absolute-ish string path suitable for chromadb.PersistentClient(path=...).
        """
        if not user_id:
            # No isolation — use shared default path
            path = self.chroma_base / "default" / "persistent.db"
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)

        if self.level == "user":
            path = self.chroma_base / "users" / user_id / "persistent.db"

        elif self.level == "session":
            if not session_id:
                raise ValueError("session_id required for 'session' scope level")
            path = self.chroma_base / "users" / user_id / "sessions" / f"{session_id}.db"
            # Register session in registry
            self._register_session(user_id, session_id)

        elif self.level == "project":
            if not project_id:
                raise ValueError("project_id required for 'project' scope level")
            path = (
                self.chroma_base / "projects" / project_id
                / "users" / user_id / "persistent.db"
            )

        else:
            raise ValueError(f"Unknown scope level: {self.level!r}")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    # ── Session Registry ─────────────────────────────────────────────────────

    def _registry_path(self, user_id: str) -> Path:
        """Path to registry.json for a given user."""
        return self.chroma_base / "users" / user_id / "registry.json"

    def _load_registry(self, user_id: str) -> dict[str, str]:
        """Load {session_id: created_at_iso} from registry.json."""
        rpath = self._registry_path(user_id)
        if rpath.exists():
            try:
                return json.loads(rpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_registry(self, user_id: str, registry: dict[str, str]):
        """Persist registry.json."""
        rpath = self._registry_path(user_id)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(
            json.dumps(registry, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _register_session(self, user_id: str, session_id: str):
        """Add a session entry if it doesn't already exist."""
        rpath = self._registry_path(user_id)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(rpath.with_suffix(".lock")))
        with lock:
            registry = self._load_registry(user_id)
            if session_id not in registry:
                registry[session_id] = datetime.now(timezone.utc).isoformat()
                self._save_registry(user_id, registry)

    # ── Session Lifecycle ────────────────────────────────────────────────────

    def list_sessions(self, user_id: str) -> list[SessionInfo]:
        """
        Return all session db files for a user with created-at timestamp.
        """
        registry = self._load_registry(user_id)
        sessions_dir = self.chroma_base / "users" / user_id / "sessions"
        results: list[SessionInfo] = []

        for sid, created_at in registry.items():
            db_path = sessions_dir / f"{sid}.db"
            size = _dir_size(db_path) if db_path.exists() else 0
            results.append(SessionInfo(
                session_id=sid,
                created_at=created_at,
                db_path=str(db_path),
                size_bytes=size,
            ))

        # Sort newest first
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Hard-delete a session db file and remove from registry.
        Returns True if a file was actually removed.
        """
        db_path = (
            self.chroma_base / "users" / user_id / "sessions" / f"{session_id}.db"
        )
        deleted = False
        if db_path.exists():
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
            deleted = True

        # Remove from registry
        registry = self._load_registry(user_id)
        if session_id in registry:
            del registry[session_id]
            self._save_registry(user_id, registry)
            deleted = True

        return deleted

    def archive_session(self, user_id: str, session_id: str) -> str:
        """
        Soft-delete: move session db to .sumo_db/archive/{user_id}/{session_id}.db.
        Returns the archive path.
        """
        db_path = (
            self.chroma_base / "users" / user_id / "sessions" / f"{session_id}.db"
        )
        archive_dir = self.chroma_base / "archive" / user_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{session_id}.db"

        if db_path.exists():
            shutil.move(str(db_path), str(archive_path))

        # Remove from active registry
        registry = self._load_registry(user_id)
        if session_id in registry:
            del registry[session_id]
            self._save_registry(user_id, registry)

        return str(archive_path)

    def cleanup_expired(self, user_id: str, ttl_hours: float = 24) -> list[str]:
        """
        Auto-delete sessions older than TTL.
        Returns list of deleted session_ids.
        """
        registry = self._load_registry(user_id)
        now = time.time()
        cutoff = now - (ttl_hours * 3600)

        expired: list[str] = []
        for sid, created_at in list(registry.items()):
            try:
                ts = datetime.fromisoformat(created_at).timestamp()
            except (ValueError, TypeError):
                continue
            if ts < cutoff:
                expired.append(sid)

        for sid in expired:
            self.delete_session(user_id, sid)

        return expired

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self, user_id: str) -> ScopeStats:
        """
        Return total collections, total disk usage in MB for a user.
        """
        user_dir = self.chroma_base / "users" / user_id
        total_bytes = _dir_size(user_dir) if user_dir.exists() else 0
        registry = self._load_registry(user_id)

        return ScopeStats(
            user_id=user_id,
            total_sessions=len(registry),
            total_disk_mb=round(total_bytes / (1024 * 1024), 2),
            session_ids=list(registry.keys()),
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _dir_size(path: Path) -> int:
    """Recursively compute total size in bytes of a directory or file."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.is_dir():
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    return total
