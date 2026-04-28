<div align="center">
  <h1>🚀 SumoSpace</h1>
  <p><b>A modular, auto-everything multi-agent execution environment.</b></p>
  <p>Zero API keys required. Runs fully locally out of the box.</p>
</div>

---

SumoSpace is a framework for building autonomous agents, multi-agent committees, and RAG pipelines that prioritize **local inference**, **data privacy**, and **pluggability**. 

Whether you need a simple CLI chat assistant, a code-aware semantic search engine, or a fully autonomous desktop-controlling agent, SumoSpace provides the orchestration layer.

## ✨ Core Features

- **Zero API Keys Default**: Powered by local HuggingFace or Ollama models. (Gemini, OpenAI, and Anthropic supported via opt-in).
- **Multi-Tenant Scope Management**: Built-in `ScopeManager` isolates data by `user_id`, `session_id`, or `project_id` on disk.
- **Memory & RAG**: `MemoryManager` and `UniversalIngestor` turn any directory into a semantic memory space with zero setup.
- **Multi-Agent Deliberation**: Built-in Planner, Critic, and Executor committees for safe, autonomous tool execution.
- **Rich Tool Ecosystem**: Filesystem, Shell, Web Search, Browser Automation, and Desktop Automation (pyautogui/OCR) tools ready to use.

---

## ⚡ Quick Start — Zero API Keys

```bash
# Install core (local inference, no API keys needed)
pip install -e "."

# Run a task immediately
sumo run "List all Python files in the current directory"

# Interactive chat
sumo chat

# Ingest codebase for RAG-enhanced answers
sumo ingest ./src
sumo run "How does the authentication module work?"
```

---

## 📦 Installation Options

SumoSpace is highly modular. Install only what you need.

```bash
# Core only — local HF + local embeddings (default)
pip install sumospace

# + Code-aware parsing (tree-sitter AST)
pip install sumospace[code]

# + PDF & Audio (Whisper) support
pip install sumospace[pdf,audio]

# + Browser & Desktop automation
pip install sumospace[browser]
pip install pyautogui pillow pytesseract # for desktop tools

# Everything local, no API keys
pip install sumospace[all]

# Cloud providers (opt-in, requires API keys)
pip install sumospace[gemini]
pip install sumospace[openai]
pip install sumospace[anthropic]
```

---

## 💻 Python API

SumoSpace is built to be embedded in your own applications.

### Basic Autonomous Agent
```python
import asyncio
from sumospace.kernel import SumoKernel, KernelConfig

async def main():
    # Zero config — local HF, local embeddings, no API key
    kernel = SumoKernel()
    async with kernel:
        # Ingest codebase for context-aware answers
        await kernel.ingest("./src")

        # Run an autonomous task
        trace = await kernel.run("Refactor the auth module in src/auth.py")
        print(trace.final_answer)

asyncio.run(main())
```

### With Scoped Memory (Multi-Tenancy)
For web servers or multi-user environments, isolate data effortlessly:
```python
config = KernelConfig(
    provider="ollama", 
    model="phi3:mini",
    user_id="user_123",        # Isolates to .sumo_db/users/user_123/
    session_id="chat_abc"      # Isolates memory to this specific session
)
kernel = SumoKernel(config)
```

---

## 🖥️ Desktop Agent Example

SumoSpace comes with a powerful **Desktop Agent** example that uses computer vision (OCR) and mouse/keyboard automation to control your computer autonomously.

1. Navigate to the example: `cd examples/desktop_agent`
2. Install desktop dependencies: `sudo apt install tesseract-ocr && pip install pyautogui pillow pytesseract streamlit`
3. Run the visual UI:
```bash
streamlit run app.py
```
*The agent will take screenshots, read the screen, plan actions, and execute clicks/typing live in the UI.*

---

## 🧠 Default Model Stack (Free & Local)

| Component | Default | Size | Notes |
|-----------|---------|------|-------|
| LLM | `microsoft/Phi-3-mini-4k-instruct` | ~2.3GB | Downloads once, cached |
| Embeddings | `BAAI/bge-base-en-v1.5` | ~440MB | Via sentence-transformers |
| Vector DB | ChromaDB | — | Local file, no server |
| Intent Classifier | Rule engine + NLI model | ~180MB | Zero API calls |
| Audio | `openai-whisper` (local) | ~145MB | NOT the OpenAI API |

### Using Ollama (Lighter RAM)
If you prefer running models via Ollama:
```bash
ollama pull phi3:mini
sumo run "Fix the bug" --provider ollama --model phi3:mini
```

---

## 🏗️ Architecture

```text
sumo run "task"
    │
    ├── [Classifier] 3-stage intent classification
    │       ├── Rule-based (~0ms)
    │       ├── Zero-shot NLI (~50-200ms, local)
    │       └── LLM fallback
    │
    ├── [ScopeManager] Multi-tenant isolation
    │       └── Resolves user/session specific ChromaDB paths
    │
    ├── [RAG & Memory]
    │       ├── Vector search (ChromaDB, cosine)
    │       └── Episodic conversation memory
    │
    ├── [Committee] Multi-agent deliberation
    │       ├── Planner → proposes execution plan
    │       ├── Critic  → reviews for risks
    │       └── Resolver → synthesises approved plan
    │
    └── [ToolRegistry] Execution
            ├── FileSystem / Shell
            ├── WebSearch / Browser
            └── Desktop (GUI Automation)
```

---

## 📊 System Requirements

| Configuration | RAM | VRAM | Storage |
|---|---|---|---|
| CPU Only (Ollama / TinyLlama) | 4GB | 0 | ~2GB |
| Default Local HF (Phi-3-mini) | 8GB | 4GB | ~4GB |
| Capable Local HF (Mistral 7B)| 16GB | 8GB | ~6GB |

**Total API keys required for full default functionality: 0**

---

## 🛠️ Development

```bash
make install-dev   # Install + dev tools
make test          # Full test suite
make lint          # Ruff linting
make fmt           # Auto-format
make info          # Show installed capabilities
```

## License

MIT © Omdeep
# SumoSpace
