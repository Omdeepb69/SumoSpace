import asyncio
import time
import os
from pathlib import Path
from sumospace.settings import SumoSettings
from sumospace.kernel import SumoKernel
from sumospace.snapshots import SnapshotManager
from sumospace.tools import WriteFileTool

async def verify_rollback():
    print("=== 1. Verifying Rollback ===")
    fpath = Path("test_rollback.py")
    fpath.write_text("def add(a, b): return a + b\n")
    
    settings = SumoSettings(workspace=".")
    mgr = SnapshotManager(settings)
    run_id = "test_rollback_123"
    
    tool = WriteFileTool(snapshot_manager=mgr, run_id=run_id)
    await tool.run("test_rollback.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    
    print("Content after modification:")
    print(fpath.read_text())
    
    mgr.rollback(run_id)
    
    print("Content after rollback:")
    print(fpath.read_text())
    
    assert "return a + b" in fpath.read_text()
    assert "int" not in fpath.read_text()
    fpath.unlink()
    print("Rollback verified.\n")

async def verify_faiss():
    print("=== 2. Verifying FAISS Persistence ===")
    settings = SumoSettings(vector_store="faiss", workspace=".")
    
    # Session 1
    print("Session 1: Adding documents")
    from sumospace.vectorstores import get_vector_store
    from sumospace.vectorstores.base import VectorDocument
    store = get_vector_store(settings)
    docs = [
        VectorDocument(id="doc1", text="authentication flow", embedding=[0.5, 0.5], metadata={}),
    ]
    await store.add_documents(docs)
    
    # Session 2
    print("Session 2: Searching without adding")
    store2 = get_vector_store(settings)
    results = await store2.search([0.5, 0.5], top_k=1)
    print(f"Results found: {len(results)}")
    assert len(results) > 0
    print("FAISS Persistence verified.\n")

async def verify_watch():
    print("=== 3. Verifying Watchdog polling ===")
    content = Path("sumospace/cli.py").read_text()
    if "from watchdog.observers import Observer" in content:
        print("Using OS-native Observer. Verified.\n")
    elif "PollingObserver" in content:
        print("WARNING: Using PollingObserver!\n")
    else:
        print("Observer import not found.\n")

async def main():
    await verify_rollback()
    await verify_faiss()
    await verify_watch()

if __name__ == "__main__":
    asyncio.run(main())
