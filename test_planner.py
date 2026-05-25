import asyncio
from sumospace.providers.gemini import GeminiProvider
import json

schema = {
    "type": "object",
    "properties": {
        "thoughts": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["action", "target", "description"]
            }
        }
    },
    "required": ["thoughts", "steps"]
}

async def main():
    provider = GeminiProvider(model="gemini-3.1-flash-lite")
    await provider.initialize()
    
    res = await provider.complete_structured(
        user="Write a python function to compute the 10th fibonacci number.",
        system="You are a planner. Return JSON.",
        schema=schema
    )
    print("RAW:", res)
    try:
        data = json.loads(res)
        print("PARSED:", data)
    except Exception as e:
        print("PARSE ERROR:", e)

asyncio.run(main())
