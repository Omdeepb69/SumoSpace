"""
examples/repo_manager.py
=========================
Demonstrates SumoKernel for repository management tasks.
Runs with ZERO API keys — uses HuggingFace Phi-3-mini locally.
"""

import asyncio
from sumospace.kernel import SumoKernel, KernelConfig


async def main():
    config = KernelConfig(
        provider="hf",        # Default local provider — no API key needed
        model="default",      # Resolves to Phi-3-mini; swap to "capable" for Mistral 7B
        workspace=".",        # Set to your actual repo path
        dry_run=True,         # Set False to actually execute
        verbose=True,
        require_consensus=True,
    )

    tasks = [
        "List all Python files in the current directory",
        "Find all TODO and FIXME comments in the codebase",
        "Check if there is a pyproject.toml and show its dependencies",
    ]

    async with SumoKernel(config) as kernel:
        # Optionally ingest the codebase for RAG-enhanced answers
        # await kernel.ingest(".")

        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print("=" * 60)

            trace = await kernel.run(task)

            print(f"\nIntent:  {trace.intent.value}")
            print(f"Success: {trace.success}")
            print(f"Steps:   {len(trace.step_traces)}")
            print(f"\nResult:\n{trace.final_answer}")

            if trace.error:
                print(f"\nError: {trace.error}")


if __name__ == "__main__":
    asyncio.run(main())
