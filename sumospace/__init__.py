# sumospace/__init__.py

"""
Sumospace — Auto-Everything Multi-Agent Execution Environment

Quick start (zero API keys required):
    from sumospace.kernel import SumoKernel, KernelConfig
    import asyncio

    async def main():
        kernel = SumoKernel()  # Zero config — local HF, local embeddings, no API key
        async with kernel:
            trace = await kernel.run("Refactor the auth logic in src/auth.py")
            print(trace.final_answer)

    asyncio.run(main())

With Ollama (also zero API key):
    kernel = SumoKernel(KernelConfig(provider="ollama", model="phi3:mini"))

With cloud provider (requires API key + pip extra):
    kernel = SumoKernel(KernelConfig(provider="gemini", model="gemini-1.5-flash"))
"""

__version__ = "0.1.0"
__author__ = "Omdeep"

# Intentionally minimal — no heavy imports at top level.
# Users import subsystems directly:
#   from sumospace.kernel import SumoKernel
#   from sumospace.ingest import UniversalIngestor
#   from sumospace import multimodal  # only if pip install sumospace[audio/video]
