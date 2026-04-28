"""
examples/code_refactor.py
==========================
Demonstrates SumoKernel for code refactoring and analysis.
Uses local HuggingFace code model — zero API keys.
"""

import asyncio
from pathlib import Path
from sumospace.kernel import SumoKernel, KernelConfig


SAMPLE_CODE = '''
# sample_module.py  — intentionally messy for demo purposes
import os,sys
from pathlib import Path
import json,re

def   do_stuff( x,y,z=None ):
    # TODO: add validation
    if x == None:
        return None
    result=[]
    for i in range(0,len(x)):
        if x[i] > y:
            result.append(x[i])
    return result

class DataHandler:
    def __init__(self):
        self.data=[]
        self.count=0
    def add(self,item):
        self.data.append(item)
        self.count=self.count+1
    def get_all(self):
        return self.data
    def process(self):
        # TODO: implement
        pass
'''


async def main():
    # Write sample messy code to disk
    sample_path = Path("/tmp/sample_module.py")
    sample_path.write_text(SAMPLE_CODE)
    print(f"Created sample module at {sample_path}")

    config = KernelConfig(
        provider="hf",
        model="code",      # Resolves to Qwen2.5-Coder — best local model for code tasks
        workspace="/tmp",
        dry_run=True,      # Set False to actually apply refactoring
        verbose=True,
    )

    tasks = [
        f"Review the code quality in {sample_path} and identify issues",
        f"Refactor {sample_path}: fix formatting, replace None checks with is None, "
        f"add type hints, and improve the loop in do_stuff to use list comprehension",
        f"Write pytest tests for the DataHandler class in {sample_path}",
    ]

    async with SumoKernel(config) as kernel:
        # Ingest the file for RAG context
        chunks = await kernel.ingest(str(sample_path))
        print(f"\nIngested {chunks} chunks from sample module")

        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task[:80]}...")
            print("=" * 60)

            trace = await kernel.run(task)

            print(f"\nIntent:  {trace.intent.value}")
            print(f"Success: {trace.success}")
            print(f"\nResult:\n{trace.final_answer[:800]}")


if __name__ == "__main__":
    asyncio.run(main())
