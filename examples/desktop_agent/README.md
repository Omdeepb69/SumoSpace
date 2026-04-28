# 🖥️ SumoSpace Desktop Agent

**Autonomous desktop automation powered by SumoSpace multi-agent orchestration.**

The agent takes a natural language task (e.g., *"Open VS Code and open the sumospace project"*),
then autonomously captures screenshots, reasons about the screen, and executes mouse/keyboard
actions — all orchestrated by SumoSpace's provider, memory, and tool systems.

## What This Showcases

| SumoSpace Feature      | How It's Used                                                |
|------------------------|--------------------------------------------------------------|
| **ProviderRouter**     | LLM brain — plans actions from screenshots                   |
| **MemoryManager**      | Tracks full action history across steps                      |
| **ScopeManager**       | Per-session isolation (each task gets its own scope)          |
| **ToolRegistry**       | Custom desktop tools registered alongside built-in tools     |
| **UniversalIngestor**  | Ingests OCR text from screenshots for RAG recall             |

## Requirements

```bash
pip install sumospace
pip install pyautogui pillow streamlit

# For OCR-based screen reading (optional but recommended)
sudo apt install tesseract-ocr
pip install pytesseract
```

## Provider Setup

Pick **any** SumoSpace provider:

```bash
# Option 1: Ollama (free, local, no API key)
ollama pull llava        # vision model
ollama pull phi3:mini    # text model

# Option 2: Gemini (free tier, vision-capable)
export GOOGLE_API_KEY=your_key

# Option 3: HuggingFace (fully offline)
# No setup needed — default
```

## Run

```bash
# Streamlit UI
streamlit run examples/desktop_agent/app.py

# CLI mode
python examples/desktop_agent/agent.py "Open the file manager and navigate to Documents"
```

## Architecture

```
User Task → ProviderRouter (LLM) → Plan Next Action
                ↓
        ToolRegistry.execute()
        ├── take_screenshot  → PIL grab
        ├── click_at         → pyautogui.click
        ├── type_text        → pyautogui.write
        ├── hotkey           → pyautogui.hotkey
        ├── read_screen      → pytesseract OCR
        ├── open_app         → subprocess
        └── shell            → (built-in SumoSpace tool)
                ↓
        MemoryManager.add() → stores action + result
                ↓
        Screenshot → OCR → next iteration
```
