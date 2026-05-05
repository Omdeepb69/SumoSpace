# sumospace/snapshots.py
"""
Snapshot + Rollback Safety
==========================
Takes file snapshots before any mutating tool operation.
Stores diffs, file contents, and run metadata in .sumo_db/snapshots/.
Integrates with audit log run IDs for cross-referencing.

CLI:
    sumo snapshots list
    sumo snapshots show <run-id>
    sumo rollback <run-id>
"""
from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileSnapshot:
    path: str
    content_before: str     # original content (empty string if file didn't exist)
    content_after: str      # content after mutation (empty string if file was deleted)
    diff: str               # unified diff
    existed_before: bool


@dataclass
class Snapshot:
    run_id: str
    timestamp: float
    files: list[FileSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def datetime_str(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "metadata": self.metadata,
            "files": [
                {
                    "path": f.path,
                    "existed_before": f.existed_before,
                    "diff": f.diff,
                }
                for f in self.files
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, contents_dir: Path) -> "Snapshot":
        files = []
        for fd in data.get("files", []):
            before_path = contents_dir / "before" / fd["path"].lstrip("/")
            after_path  = contents_dir / "after"  / fd["path"].lstrip("/")
            files.append(FileSnapshot(
                path=fd["path"],
                content_before=before_path.read_text(encoding="utf-8", errors="replace") if before_path.exists() else "",
                content_after=after_path.read_text(encoding="utf-8", errors="replace") if after_path.exists() else "",
                diff=fd.get("diff", ""),
                existed_before=fd.get("existed_before", True),
            ))
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            files=files,
            metadata=data.get("metadata", {}),
        )


class SnapshotManager:
    """
    Manages file snapshots for safe rollback.

    Usage:
        mgr = SnapshotManager(settings)
        run_id = "abc123"

        # Before mutating a file:
        mgr.snapshot_file(run_id, "/path/to/file.py")

        # After the run, to rollback:
        mgr.rollback(run_id)
    """

    def __init__(self, settings):
        self._settings = settings
        self._base = Path(settings.chroma_base) / "snapshots"
        self._base.mkdir(parents=True, exist_ok=True)
        # In-memory buffer: run_id -> list of snapshotted paths (to avoid double-snapshotting)
        self._snapshotted: dict[str, set[str]] = {}

    def _run_dir(self, run_id: str) -> Path:
        return self._base / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def _before_path(self, run_id: str, file_path: str) -> Path:
        return self._run_dir(run_id) / "before" / Path(file_path).name

    def _after_path(self, run_id: str, file_path: str) -> Path:
        return self._run_dir(run_id) / "after" / Path(file_path).name

    def _load_manifest(self, run_id: str) -> dict:
        mp = self._manifest_path(run_id)
        if mp.exists():
            return json.loads(mp.read_text())
        return {
            "run_id": run_id,
            "timestamp": time.time(),
            "metadata": {},
            "files": [],
        }

    def _save_manifest(self, run_id: str, manifest: dict):
        mp = self._manifest_path(run_id)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=2))

    def snapshot_file(self, run_id: str, file_path: str) -> bool:
        """
        Capture file state BEFORE mutation.
        Must be called before any write/patch operation.
        Returns True if snapshot was taken, False if already snapshotted this run.
        """
        path = Path(file_path).resolve()
        key = str(path)

        if run_id not in self._snapshotted:
            self._snapshotted[run_id] = set()

        if key in self._snapshotted[run_id]:
            return False  # Already snapshotted in this run

        self._snapshotted[run_id].add(key)
        existed = path.exists()
        content_before = path.read_text(encoding="utf-8", errors="replace") if existed else ""

        # Persist before-content
        before_dir = self._run_dir(run_id) / "before"
        before_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{path.name}_{abs(hash(key)) % 100000}"
        before_file = before_dir / safe_name
        before_file.write_text(content_before, encoding="utf-8")

        # Update manifest
        manifest = self._load_manifest(run_id)
        manifest["files"].append({
            "path": str(path),
            "before_file": safe_name,
            "existed_before": existed,
            "diff": "",  # populated after mutation
        })
        self._save_manifest(run_id, manifest)
        return True

    def record_after(self, run_id: str, file_path: str):
        """
        Capture file state AFTER mutation and compute diff.
        Call this after the write/patch operation completes.
        """
        path = Path(file_path).resolve()
        key = str(path)
        manifest = self._load_manifest(run_id)

        file_entry = next(
            (f for f in manifest["files"] if f["path"] == key), None
        )
        if not file_entry:
            return

        content_after = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

        # Save after content
        after_dir = self._run_dir(run_id) / "after"
        after_dir.mkdir(parents=True, exist_ok=True)
        after_file = after_dir / file_entry["before_file"]
        after_file.write_text(content_after, encoding="utf-8")
        file_entry["after_file"] = file_entry["before_file"]

        # Compute diff
        before_path = self._run_dir(run_id) / "before" / file_entry["before_file"]
        content_before = before_path.read_text(encoding="utf-8", errors="replace") if before_path.exists() else ""
        diff = "".join(difflib.unified_diff(
            content_before.splitlines(keepends=True),
            content_after.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        ))
        file_entry["diff"] = diff
        self._save_manifest(run_id, manifest)

    def rollback(self, run_id: str) -> list[str]:
        """
        Restore all files to their pre-run state.
        Returns list of restored file paths.
        """
        manifest = self._load_manifest(run_id)
        restored = []

        for fe in manifest.get("files", []):
            path = Path(fe["path"])
            before_file = self._run_dir(run_id) / "before" / fe.get("before_file", "")

            if not fe.get("existed_before", True):
                # File was created during this run — delete it
                if path.exists():
                    path.unlink()
                    restored.append(f"[deleted] {path}")
            elif before_file.exists():
                # Restore original content
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    before_file.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                )
                restored.append(f"[restored] {path}")

        return restored

    def list_snapshots(self) -> list[dict]:
        """Return all snapshots sorted by most recent first."""
        snapshots = []
        for run_dir in sorted(self._base.iterdir(), reverse=True):
            mp = run_dir / "manifest.json"
            if mp.exists():
                try:
                    data = json.loads(mp.read_text())
                    snapshots.append({
                        "run_id": data.get("run_id", run_dir.name),
                        "datetime": data.get("datetime", ""),
                        "timestamp": data.get("timestamp", 0),
                        "files_count": len(data.get("files", [])),
                        "metadata": data.get("metadata", {}),
                    })
                except Exception:
                    pass
        return snapshots

    def show_snapshot(self, run_id: str) -> dict | None:
        """Return full snapshot manifest for a run ID."""
        mp = self._manifest_path(run_id)
        if not mp.exists():
            return None
        return json.loads(mp.read_text())

    def delete_snapshot(self, run_id: str) -> bool:
        """Delete a snapshot. Returns True if deleted."""
        import shutil
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False
