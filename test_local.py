import asyncio
from sumospace.kernel import SumoKernel
from sumospace.settings import SumoSettings

async def run():
    settings = SumoSettings(provider="ollama", model="llama3:8b", committee_enabled=False)
    async with SumoKernel(settings=settings) as kernel:
        trace = await kernel.run("Add a docstring to def login() in auth.py using replace_text.")
        print(trace.success)
        print(trace.tool_calls)
        print(trace.error)

asyncio.run(run())
