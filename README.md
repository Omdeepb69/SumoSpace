# SumoSpace

**SumoSpace is a locally-first, multi-tenant autonomous agent framework for executing complex software tasks safely.**

## How is SumoSpace different from LangChain?

| | SumoSpace | LangChain |
|---|---|---|
| Local inference | First-class (HF, Ollama, vLLM) | Bolted on |
| Multi-user isolation | Built-in ScopeManager | DIY |
| Planning safety | 3-agent committee | Single LLM call |
| Cloud required | No | Effectively yes |
| Learning curve | Low (fixed pipeline) | High (graph mental model) |

SumoSpace is not better than LangChain for every use case. If you need 700+ integrations or fine-grained graph control, use LangChain. If you need autonomous task execution on private data without cloud dependencies, use SumoSpace.

## Quickstart

```bash
pip install sumospace
```

```python
from sumospace import SumoKernel, SumoSettings

async with SumoKernel(SumoSettings(provider="ollama", model="phi3:mini")) as kernel:
    trace = await kernel.run("Find all TODO comments in ./src and create a GitHub issue summary")
    print(trace.final_answer)
```

## Architecture Overview

Unlike traditional LLM wrappers, SumoSpace operates as an autonomous engine structured around safety and reliability.

1. **Classifier**: Determines if the incoming task requires web search, codebase retrieval (RAG), or direct execution.
2. **Committee**: A group of three LLM sub-agents (Planner, Critic, Resolver) deliberate to agree on a step-by-step execution plan *before* any action is taken.
3. **Kernel**: The orchestrator that manages tool execution, isolated memory across sessions, and hooks for real-time observability.

## Features

- **Multi-Tenant by Default**: Use `scope_level="user"` to strictly isolate file access, memory, and RAG context between different users.
- **Provider Resilience**: Automatically falls back from cloud providers (OpenAI, Gemini) to local models (HuggingFace, Ollama) on network failure or rate limits.
- **Real-Time Streaming**: Use `stream_run()` to push progress updates to frontends instantaneously.
- **Full Telemetry**: Native OpenTelemetry (OTLP) integration for deep profiling of LLM latency.

## Documentation

- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

See the `/examples` directory for advanced usage patterns, including FastAPI integration and custom telemetry hooks.
