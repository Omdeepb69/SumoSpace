"""
SumoSpace Quickstart
====================
This is the minimal setup to run an autonomous agent on your local codebase.
No API keys required. It defaults to using HuggingFace local models.
"""

import asyncio
from sumospace import SumoKernel, SumoSettings

async def main():
    # Configure the settings. 
    # By default, it uses 'hf' (HuggingFace) for local inference.
    settings = SumoSettings(
        provider="hf",
        model="default",  # Automatically selects a good default (e.g., Phi-3)
        workspace=".",    # Target directory for the agent to act on
    )

    # Initialize the kernel within an async context manager.
    # This guarantees proper lifecycle management (booting models, cleaning up locks).
    async with SumoKernel(settings=settings) as kernel:
        print("Kernel booted. Deliberating and executing...")
        
        # Pass a natural language task
        trace = await kernel.run("Find all TODO comments in ./src and summarize them.")
        
        # The trace contains the final LLM synthesis, success status, and full step history.
        if trace.success:
            print("\nFinal Answer:\n", trace.final_answer)
        else:
            print(f"\nExecution Failed: {trace.error}")

if __name__ == "__main__":
    asyncio.run(main())
