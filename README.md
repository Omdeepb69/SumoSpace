# SumoSpace

**Locally-first. Multi-agent. Zero cloud dependencies.**

SumoSpace is a production-grade autonomous agent framework that runs entirely on your machine.
No API keys required to get started. No usage fees. No data leaves your environment.

```bash
pip install sumospace
sumo run "Refactor all sync functions in src/ to async"
```

---

## Why SumoSpace

| | SumoSpace | LangChain | LlamaIndex |
|---|---|---|---|
| **Default provider** | Local (HF / Ollama) | OpenAI | OpenAI |
| **API key required** | ❌ No | ✅ Yes | ✅ Yes |
| **Multi-agent committee** | ✅ Built-in | ➕ Extra | ❌ No |
| **File snapshot + rollback** | ✅ Built-in | ❌ No | ❌ No |
| **Media RAG** | ✅ Built-in | ➕ Extra | ➕ Extra |
| **Benchmarks** | ✅ Built-in | ❌ No | ❌ No |

---

## Quickstart

```bash
pip install sumospace
sumo ingest ./src          # Index your codebase
sumo run "Add docstrings to all public functions"
sumo rollback <run-id>     # Undo if needed
```

---

## One Killer Example

```python
import asyncio
from sumospace import SumoKernel, SumoSettings

async def main():
    settings = SumoSettings.for_coding()
    async with SumoKernel(settings=settings) as kernel:
        await kernel.ingest("./src")
        trace = await kernel.run("Fix all TODO comments in src/auth.py")
        print(trace.final_output)

asyncio.run(main())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SumoKernel                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Planner  │→ │  Critic  │→ │    Resolver      │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│         Committee (optional, 3-agent)               │
├─────────────────────────────────────────────────────┤
│   RAG Pipeline    │  Memory   │  Audit Logs        │
│   (multi-query)   │  (SQLite) │  (JSON)            │
├─────────────────────────────────────────────────────┤
│   Vector Store    │  Snapshots + Rollback           │
│   chroma/faiss/   │  (.sumo_db/snapshots/)         │
│   qdrant          │                                │
├─────────────────────────────────────────────────────┤
│          Tools: file, shell, web, docker, deps     │
└─────────────────────────────────────────────────────┘
```

---

📖 **Full documentation:** [omdeepb69.github.io/SumoSpace](https://omdeepb69.github.io/SumoSpace)

## Benchmarks

Run your own benchmarks against the bundled sample project:

```bash
sumo benchmark compare --provider ollama --model phi3:mini
```

We are running standardized benchmarks across hardware configurations
and will publish results in v0.3. Community benchmark submissions welcome.
