# tests/test_snapshots.py
"""Tests for SnapshotManager."""
from __future__ import annotations

import pytest
from pathlib import Path
from sumospace.snapshots import SnapshotManager
from sumospace.settings import SumoSettings


@pytest.fixture
def mgr(tmp_path):
    settings = SumoSettings(chroma_base=str(tmp_path / ".sumo_db"))
    return SnapshotManager(settings)


def test_snapshot_new_file(mgr, tmp_path):
    """Snapshotting a new file marks it as not existed_before."""
    fpath = tmp_path / "hello.py"
    fpath.write_text("print('hello')")
    run_id = "run001"
    taken = mgr.snapshot_file(run_id, str(fpath))
    assert taken is True

    manifest = mgr.show_snapshot(run_id)
    assert manifest is not None
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["existed_before"] is True


def test_snapshot_not_doubled(mgr, tmp_path):
    """Same file is only snapshotted once per run."""
    fpath = tmp_path / "foo.py"
    fpath.write_text("x = 1")
    run_id = "run002"
    assert mgr.snapshot_file(run_id, str(fpath)) is True
    assert mgr.snapshot_file(run_id, str(fpath)) is False  # already snapshotted


def test_rollback_restores_content(mgr, tmp_path):
    """Rollback should restore original file content."""
    fpath = tmp_path / "target.py"
    original = "# original content\nx = 1\n"
    fpath.write_text(original)

    run_id = "run003"
    mgr.snapshot_file(run_id, str(fpath))

    # Mutate the file
    fpath.write_text("# mutated\nx = 999\n")
    mgr.record_after(run_id, str(fpath))

    # Rollback
    restored = mgr.rollback(run_id)
    assert len(restored) == 1
    assert fpath.read_text() == original


def test_rollback_deletes_new_file(mgr, tmp_path):
    """Rollback removes files that did not exist before the run."""
    fpath = tmp_path / "brand_new.py"
    run_id = "run004"

    # Snapshot before file exists
    taken = mgr.snapshot_file(run_id, str(fpath))

    # Update manifest to mark as not existed
    manifest = mgr.show_snapshot(run_id)
    manifest["files"][0]["existed_before"] = False
    mgr._save_manifest(run_id, manifest)

    # Create the file (simulating a write_file tool call)
    fpath.write_text("# new file created by agent\n")

    mgr.rollback(run_id)
    assert not fpath.exists()


def test_list_snapshots(mgr, tmp_path):
    fpath = tmp_path / "a.py"
    fpath.write_text("a=1")
    for run_id in ["r1", "r2", "r3"]:
        mgr.snapshot_file(run_id, str(fpath))

    snaps = mgr.list_snapshots()
    assert len(snaps) == 3


def test_show_nonexistent_snapshot(mgr):
    result = mgr.show_snapshot("does_not_exist")
    assert result is None


def test_diff_recorded_after_mutation(mgr, tmp_path):
    fpath = tmp_path / "diff_test.py"
    fpath.write_text("x = 1\n")
    run_id = "run_diff"
    mgr.snapshot_file(run_id, str(fpath))
    fpath.write_text("x = 2\n")
    mgr.record_after(run_id, str(fpath))

    manifest = mgr.show_snapshot(run_id)
    assert "-x = 1" in manifest["files"][0]["diff"]
    assert "+x = 2" in manifest["files"][0]["diff"]
