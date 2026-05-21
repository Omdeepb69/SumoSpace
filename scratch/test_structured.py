import asyncio
import httpx

async def main():
    client = httpx.AsyncClient(base_url="http://localhost:11434", timeout=60)
    payload = {
        "model": "qwen2.5-coder:3b",
        "messages": [
            {"role": "system", "content": "You are a planner. Return JSON matching the schema."},
            {"role": "user", "content": "Task: hello"}
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "tool": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["step_number", "tool", "description"]
                    }
                }
            },
            "required": ["reasoning", "steps"]
        }
    }
    try:
        r = await client.post("/api/chat", json=payload)
        print("Status code:", r.status_code)
        print("Response:", r.json())
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
