import pytest
import os
import tempfile
from pathlib import Path
from sumospace.tools import ReplaceTextTool
from sumospace.snapshots import SnapshotManager


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("def login():\n    pass\n\ndef logout():\n    pass\n")
        path = f.name
    yield path
    os.remove(path)


@pytest.mark.asyncio
async def test_replace_text_success(temp_file):
    tool = ReplaceTextTool()
    old_text = "def login():\n    pass"
    new_text = "def login():\n    \"\"\"Login docstring.\"\"\"\n    pass"
    
    res = await tool.run(temp_file, old_text, new_text)
    
    assert res.success is True
    content = Path(temp_file).read_text(encoding="utf-8")
    assert "Login docstring" in content
    assert "def logout():" in content  # Untouched


@pytest.mark.asyncio
async def test_replace_text_crlf_resilience(temp_file):
    # Simulate LLM outputting CRLF
    tool = ReplaceTextTool()
    old_text = "def login():\r\n    pass"
    new_text = "def login():\r\n    pass\r\n    return True"
    
    res = await tool.run(temp_file, old_text, new_text)
    
    assert res.success is True
    content = Path(temp_file).read_text(encoding="utf-8")
    assert "return True" in content


@pytest.mark.asyncio
async def test_replace_text_not_found(temp_file):
    tool = ReplaceTextTool()
    res = await tool.run(temp_file, "def register():", "def register():\n    pass")
    
    assert res.success is False
    assert "old_text not found" in res.error


@pytest.mark.asyncio
async def test_replace_text_multiple_matches_error():
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        # File has two identical functions
        f.write("def helper():\n    return 1\n\ndef helper():\n    return 2\n")
        path = f.name

    tool = ReplaceTextTool()
    res = await tool.run(path, "def helper():", "def helper():\n    pass")
    
    assert res.success is False
    assert "old_text matched 2 locations" in res.error
    # Check diagnostics
    assert "lines" in res.error
    assert "1" in res.error
    assert "4" in res.error
    
    os.remove(path)


@pytest.mark.asyncio
async def test_replace_text_rollback(temp_file):
    """Test that a failed write can be rolled back via the snapshot manager."""
    import shutil
    
    class DummySettings:
        chroma_base = os.path.join(os.path.dirname(temp_file), ".sumo_db")
        
    sm = SnapshotManager(settings=DummySettings())
    run_id = "test_run_123"
    
    tool = ReplaceTextTool(snapshot_manager=sm, run_id=run_id)
    
    # Pre-snapshot content
    orig_content = Path(temp_file).read_text()
    
    # 1. Do a successful replacement to trigger the before-snapshot and write
    res = await tool.run(temp_file, "def login():\n    pass", "def login():\n    BROKEN")
    assert res.success is True
    
    # 2. Simulate a mid-run failure / bad LLM logic by manually breaking the file further
    Path(temp_file).write_text("CORRUPTED")
    
    # 3. Rollback using the snapshot manager
    sm.rollback(run_id)
    
    # 4. Verify original content is restored
    restored_content = Path(temp_file).read_text()
    assert restored_content == orig_content
