# SumoFinance

A personal finance AI copilot example built exclusively with the **SumoSpace** framework. 

SumoFinance demonstrates how to use the SumoKernel and ScopeManager to create a locally-isolated, zero-dependency AI assistant. 
- **Zero external SDKs**: No direct imports of `chromadb`, `sentence-transformers`, or LLM SDKs.
- **Tools for Math**: All calculations (safe-to-spend, goal progress, savings schedules) are strictly offloaded to deterministic tools, avoiding LLM hallucinations.
- **RAG-First Context**: All user context (transactions, schedules, preferences) is semantically stored in the Vector Database (VDB). The AI reads from RAG before every turn.
- **ScopeManager Isolation**: Demonstrates handling two simultaneous paths per user: `level="user"` for long-term transaction and preference RAG data, and `level="session"` for short-term conversational chat memory.

## Architecture
- **Backend**: FastAPI (`api.py`)
- **Frontend**: Streamlit (`app.py`)
- **Database**: SQLite (`db.py`)
- **AI Core**: SumoSpace (`finance_kernel.py` + `tools.py`)

## System Requirements
- **Minimum**: 8GB RAM, no GPU needed (CPU inference ~15-30s per response).
- **Recommended**: 8GB VRAM GPU (response time ~2-3s via 4-bit quantization).
- **Auto-Acceleration**: If Ollama is running, the kernel will automatically prefer it for faster inference. On GPU, 4-bit quantization is enabled by default to fit models in ~4GB VRAM.

## Running Locally

1. **Install Requirements**
Ensure you have installed the core `sumospace` package first.
```bash
cd examples/sumofinance
pip install -r requirements.txt
```

2. **Start the FastAPI Backend**
In one terminal, run the backend server:
```bash
cd examples/sumofinance
python api.py
```
*(Alternatively: `uvicorn api:app --reload`)*

3. **Start the Streamlit Frontend**
In a second terminal, launch the Streamlit app:
```bash
cd examples/sumofinance
streamlit run app.py
```

## Usage Flow
1. Open the Streamlit app in your browser (usually `http://localhost:8501`).
2. Click **"Initialize Test User"** to create a dummy user and initialize the isolated ScopeManager `.sumo_db` directories.
3. Check out the **Dashboard** to see the initial Safe-to-Spend calculations.
4. Try out the **Mock Transaction** button on the Dashboard. This simulates an expense, saves it to SQLite, and instantly ingests it as a text chunk into the Vector DB.
5. Go to the **Goals** tab to create a new financial goal. The deterministic `SavingsScheduleTool` will compute daily/weekly/monthly targets, verify feasibility, and ingest the schedule into the VDB.
6. Head to the **AI Copilot** tab and ask the assistant: "What are my goals and do I have enough safe to spend today?". The AI will retrieve the ingested chunks to answer accurately.
