import asyncio
import sys
from pathlib import Path
import shutil
import tempfile

from sumospace.settings import SumoSettings
from sumospace.kernel import SumoKernel

FIXTURES_DIR = Path("/mnt/data/projects/sumospace/sumospace/sumospace/benchmarks/fixtures/sample_project")

async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ws = Path(tmpdir)
        shutil.copytree(FIXTURES_DIR, tmp_ws, dirs_exist_ok=True)
        
        settings = SumoSettings(
            provider="ollama",
            model="qwen2.5-coder:3b",
            workspace=str(tmp_ws),
            committee_enabled=True,
            committee_mode="full",
            vector_store="faiss",
            memory_enabled=False,
            rag_enabled=False,
            verbose=True,
            dry_run=False,
            execution_mode="react",
        )
        
        async with SumoKernel(settings=settings) as kernel:
            trace = await kernel.run("Add a docstring to every function in utils.py that is missing one. Do not change any logic.")
            print("Trace final answer:", trace.final_answer)

if __name__ == "__main__":
    asyncio.run(main())
