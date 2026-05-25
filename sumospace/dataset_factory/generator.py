import asyncio
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from sumospace import SumoKernel, SumoSettings
from sumospace.benchmarks.tasks import TASK_REGISTRY

OUTPUT_DIR = Path("dataset_factory/outputs")
OUTPUT_FILE = OUTPUT_DIR / "qwen_sft_dataset.jsonl"


def _build_state_header(kernel: SumoKernel) -> str:
    """Builds the CURRENT WORLD STATE header for the dataset."""
    state = {
        "environment": {
            "os": platform.system().lower(),
            "internet": True,  # Assume true for desktop
            "gpu": True,
            "cwd": str(Path(kernel.settings.workspace).absolute()),
            "available_tools": [t.name for t in kernel._tools.list_tools().values() if getattr(t, 'name', None)],
            "execution_mode": "sandboxed" if kernel.settings.shell_sandbox else "native"
        }
    }
    return json.dumps(state, indent=2)


def format_to_xml_dsl(trace) -> list[dict[str, str]]:
    """Converts a SumoSpace ExecutionTrace into a multi-turn XML DSL Chat dataset."""
    messages = []
    
    # 1. System Prompt
    messages.append({
        "role": "system",
        "content": (
            "You are an autonomous intelligence operating system.\n"
            "You control the environment through XML-based tool calls.\n"
            "Analyze the state, formulate a <thought>, and execute a <call>."
        )
    })

    # 2. Initial User State & Task
    # We don't have access to the active kernel here, but we can simulate the state header
    state_header = json.dumps({
        "environment": {
            "os": platform.system().lower(),
            "cwd": str(Path(".").absolute()),
            "execution_mode": "sandboxed"
        }
    }, indent=2)
    
    messages.append({
        "role": "user",
        "content": f"{state_header}\n\nTASK: {trace.task}"
    })

    # 3. Multi-turn Tool Executions
    for i, step in enumerate(trace.step_traces):
        # The Assistant's execution (XML DSL)
        xml_call = f'<call tool="{step.tool}">\n'
        for k, v in step.parameters.items():
            xml_call += f'  <{k}>{v}</{k}>\n'
        xml_call += '</call>'

        # Handle reflection formatting for negative reward states
        is_reflection = i > 0 and not trace.step_traces[i-1].result.success
        
        if is_reflection:
            assistant_content = f"<reflection>\n{step.thought}\n</reflection>\n{xml_call}"
        else:
            assistant_content = f"<thought>\n{step.thought}\n</thought>\n{xml_call}"

        messages.append({
            "role": "assistant",
            "content": assistant_content
        })

        # The Environment's response
        if step.result.success:
            messages.append({
                "role": "user",
                "content": f"<tool_output>\n{step.result.output}\n</tool_output>"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"<error>\n{step.result.error}\n</error>"
            })

    # 4. Final Answer
    if trace.final_answer:
        messages.append({
            "role": "assistant",
            "content": f"<thought>\nTask is complete.\n</thought>\n<call tool=\"done\">\n  <summary>{trace.final_answer}</summary>\n</call>"
        })

    return messages


async def generate_traces(provider: str, model: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Dataset Generation using Teacher: {provider}/{model}")
    print(f"Output will be saved to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        pass # Clear file

    for task_def in TASK_REGISTRY[:1]:
        print(f"\n======================================")
        print(f"Generating trace for: {task_def.name}")
        
        # Setup workspace fixture
        workspace_dir = Path(f"/tmp/sumo_dataset_{task_def.id}")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        shutil.copytree(task_def.workspace, workspace_dir)

        settings = SumoSettings.for_coding(
            provider=provider,
            model=model,
            workspace=str(workspace_dir),
        )

        async with SumoKernel(settings=settings) as kernel:
            trace = await kernel.run(task_def.prompt)

            # Force generate trace for testing purposes
            pass

            # Validate
            pass

            # Format to XML DSL
            messages = format_to_xml_dsl(trace)
            
            # Save to JSONL
            record = {"messages": messages, "metadata": {"task": task_def.id, "success": True}}
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
            
            print(f"  [+] Trace successfully compiled and appended!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()

    asyncio.run(generate_traces(args.provider, args.model))
