import asyncio
import os
from sumospace.committee import PlannerAgent
from sumospace.providers import OllamaProvider

async def run():
    provider = OllamaProvider("llama3:8b")
    await provider.connect()
    try:
        planner = PlannerAgent(provider)
        
        # Simulate the prompt
        task = "Add Google-style docstrings to all public functions in auth.py, database.py, and utils.py. Do not change any logic. Only add docstrings."
        context = "AVAILABLE TOOLS:\nreplace_text\nread_file"
        
        # We set this to see the raw output
        os.environ["DEBUG_PLANNER"] = "1"
        plan, raw_clean = await planner.plan(task, context)
        print("Plan steps:", len(plan.steps))
        print("Reasoning:", plan.reasoning)
        if not plan.steps:
            print("Raw output:", plan.raw_output)
    finally:
        await provider.disconnect()

asyncio.run(run())
