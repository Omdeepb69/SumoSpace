import asyncio
from sumospace.settings import SumoSettings
from sumospace.kernel import SumoKernel
from sumospace.tools import BaseTool, ToolResult
from sumospace.hooks import HookRegistry

async def run_section7():
    class ReverseStringTool(BaseTool):
        name = "reverse_string"
        description = "Reverses a given string."
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        async def run(self, text: str, **kwargs) -> ToolResult:
            return ToolResult(tool=self.name, success=True, output=text[::-1])

    class CountWordsTool(BaseTool):
        name = "count_words"
        description = "Counts the number of words in a given text string. Parameter: text (string, required)."
        schema = {"type": "object", "properties": {"text": {"type": "string", "description": "The text string to count words in."}}, "required": ["text"]}
        param_aliases = {"content": "text", "string": "text", "input": "text", "data": "text"}
        async def run(self, text: str, **kwargs) -> ToolResult:
            return ToolResult(tool=self.name, success=True, output=str(len(text.split())))

    tool1 = ReverseStringTool()
    tool2 = CountWordsTool()
    
    settings = SumoSettings(provider="ollama", model="phi3:mini", vector_store="faiss", verbose=True)
    async with SumoKernel(settings=settings) as kernel:
        kernel.tools.register(tool1)
        kernel.tools.register(tool2)
        
        trace = await kernel.run("Use the count_words tool to count the words in the string 'The quick brown fox'.")
        
        used_tools = [s.tool for s in getattr(trace, 'step_traces', [])]
        print(f"\nTrace final answer: {trace.final_answer}")
        print(f"Tools used in step_traces: {used_tools}")

asyncio.run(run_section7())
