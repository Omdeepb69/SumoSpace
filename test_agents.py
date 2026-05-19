import asyncio
from sumospace.providers.ollama import OllamaProvider
from sumospace.committee import PlannerAgent, CriticAgent, ResolverAgent
from sumospace.templates import TemplateManager

async def main():
    provider = OllamaProvider(model="qwen2.5-coder:3b")
    await provider.initialize()
    
    planner = PlannerAgent(provider)
    critic = CriticAgent(provider)
    resolver = ResolverAgent(provider)
    
    task = "Add docstrings to all functions in math_ops.py"
    print("--- PLANNER ---")
    plan, raw = await planner.plan(task)
    print("Plan steps:", len(plan.steps))
    print("Raw planner output:\n", raw)
    
    print("\n--- CRITIC ---")
    verdict, reason, risks, blockers, raw_crit = await critic.critique(plan, task)
    print("Verdict:", verdict)
    print("Reason:", reason)
    print("Blockers:", blockers)
    print("Raw critic output:\n", raw_crit)
    
    print("\n--- RESOLVER ---")
    res_plan, approved, notes, raw_res = await resolver.resolve(task, plan, verdict, reason, risks, blockers)
    print("Approved:", approved)
    print("Notes:", notes)
    print("Raw resolver output:\n", raw_res)

asyncio.run(main())
