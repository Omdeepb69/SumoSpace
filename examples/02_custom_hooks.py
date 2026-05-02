"""
SumoSpace Custom Hooks
======================
This example demonstrates how to use the event-driven hooks system.
Hooks are perfect for streaming UI updates to a web frontend or CLI dashboard.
"""

import asyncio
from sumospace import SumoKernel, SumoSettings
from sumospace.hooks import HookRegistry
from sumospace.committee import CommitteeVerdict
from sumospace.kernel import ExecutionTrace

async def main():
    # 1. Create a HookRegistry and define your event listeners
    hooks = HookRegistry()

    @hooks.on("on_task_start")
    async def task_started(task: str, session_id: str):
        print(f"\n[UI] Task started: {task} (Session: {session_id})")

    @hooks.on("on_plan_approved")
    async def plan_approved(plan, verdict: CommitteeVerdict):
        print(f"[UI] Committee approved a plan with {len(plan.steps)} steps.")
        print(f"[UI] Estimated time: {plan.estimated_duration_s}s")

    @hooks.on("on_step_start")
    async def step_started(step_number: int, tool: str, desc: str):
        print(f"[UI] 🔄 Step {step_number}: {desc} [{tool}]")

    @hooks.on("on_step_complete")
    async def step_completed(step_number: int, result, duration_ms: float):
        status = "✅" if result.success else "❌"
        print(f"[UI] {status} Step {step_number} finished in {duration_ms:.0f}ms")

    @hooks.on("on_task_complete")
    async def task_completed(trace: ExecutionTrace):
        print(f"\n[UI] 🎉 Task complete! Total time: {trace.duration_ms / 1000:.1f}s")

    # 2. Attach hooks to the kernel
    settings = SumoSettings(provider="hf")
    async with SumoKernel(settings=settings, hooks=hooks) as kernel:
        
        # 3. Use stream_run() to yield control back to the event loop incrementally
        print("Starting stream...")
        async for event in kernel.stream_run("List all python files in the current directory."):
            # Events yielded here can also be used to drive a UI,
            # but hooks are often cleaner for centralized logging/metrics.
            pass

if __name__ == "__main__":
    asyncio.run(main())
