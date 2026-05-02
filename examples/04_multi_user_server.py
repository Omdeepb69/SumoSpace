"""
SumoSpace Multi-User Server (FastAPI)
=====================================
This example demonstrates how to deploy SumoSpace in a production API.
Crucially, it shows how to isolate users and properly manage kernel lifecycles.

WARNING: Never share a single `SumoKernel` instance across concurrent requests.
Always instantiate the kernel per-request inside an `async with` block.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sumospace import SumoKernel, SumoSettings

# FastAPI App setup
app = FastAPI(title="SumoSpace API")

# Request Model
class RunRequest(BaseModel):
    user_id: str
    task: str

@app.post("/run")
async def run_task(request: RunRequest):
    """
    Execute a task for a specific user.
    """
    
    # 1. Define settings scoped strictly to this user
    settings = SumoSettings(
        provider="hf",            # Or "openai", "ollama", etc.
        user_id=request.user_id,  # Isolates memory and logs
        scope_level="user",       # Ensures this user cannot access other users' data
        workspace=".",
    )

    # 2. Instantiate the kernel per-request using `async with`.
    # This guarantees that `kernel.boot()` is called before execution,
    # and that resources/file locks are released via `kernel.shutdown()` 
    # even if an exception occurs mid-run.
    async with SumoKernel(settings=settings) as kernel:
        try:
            # 3. Execute the task
            trace = await kernel.run(request.task)
            
            return {
                "success": trace.success,
                "answer": trace.final_answer,
                "error": trace.error if not trace.success else None,
                "duration_ms": trace.duration_ms,
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run: python 04_multi_user_server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
