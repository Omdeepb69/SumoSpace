"""SumoFinance Kernel — Direct RAG→LLM chat with intent-based tool calling."""
import re
import uuid
import tempfile
import os
from datetime import datetime
from typing import Optional
from sumospace.kernel import SumoKernel, KernelConfig
from sumospace.scope import ScopeManager
from sumospace.memory import MemoryManager
import tools
import db
import chromadb
from chromadb.utils import embedding_functions

# ─── System Prompt ────────────────────────────────────────────────────────────

FINANCE_SYSTEM_PROMPT = """You are Sumo, a personal finance AI advisor.
You have DIRECT access to the user's real financial data shown in the context.

RULES:
- ALWAYS use the ACTUAL numbers from the data provided — NEVER guess or hallucinate.
- Format all currency as ₹X,XXX (Indian Rupees).
- Be actionable and specific. No generic advice like "save more money".
- Reference specific merchants, amounts, and dates from the data.
- RESPONSE LENGTH: Keep your total response between 100 to 150 words maximum.
- If you don't have enough data to answer, say so honestly.
- If the user asks to perform an action (like creating a goal), tell them it has already been done if it was handled by the system.
"""

# ─── Intent Detection (rule-based, ~0ms) ─────────────────────────────────────

INTENT_PATTERNS = [
    (r"\b(?:mark|set|change|make)\b.+\b(?:essential|avoidable)\b", "update_merchant"),
    (r"\b(?:cancel|refuse|stop|don'?t)\b.+\b(?:recommend|suggest)", "refuse_suggestion"),
    (r"\b(?:change|update|extend|move)\b.+\b(?:goal|deadline|target)", "update_goal"),
    (r"\b(?:add|create|set|make|start)\b.+\b(?:goal|target|fund)\b", "create_goal"),
    (r"\b(?:add|create|set)\b.+\b(?:fixed bill|recurring|subscription)", "add_bill"),
    (r"\b(?:save|increase saving|put aside)\b.+₹?\d", "savings_schedule"),
    (r"\b(?:pause|stop|freeze)\b.+\b(?:goal)", "pause_goal"),
    (r"\b(?:unpause|resume|restart)\b.+\b(?:goal)", "unpause_goal"),
    (r"\b(?:where|how|what)\b.+\b(?:invest|put|surplus|schemes)\b", "discover_schemes"),
    (r"\b(?:add|mock|record|log)\b.+\b(?:transaction|spend|spent|expense|payment)\b", "mock_transaction"),
]

def detect_intent(message: str) -> Optional[str]:
    """Fast regex intent detection. Returns action name or None for query."""
    msg = message.lower()
    for pattern, action in INTENT_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return action
    return None

def extract_amount(msg: str) -> Optional[float]:
    """Extract ₹ amount from message."""
    m = re.search(r'₹?\s*(\d[\d,]*(?:\.\d+)?)', msg)
    if m:
        return float(m.group(1).replace(',', ''))
    return None

def extract_merchant(msg: str) -> Optional[str]:
    """Extract merchant name — word after 'at' or quoted string."""
    m = re.search(r"['\"]([^'\"]+)['\"]", msg)
    if m: return m.group(1)
    m = re.search(r'\bat\s+(\w+)', msg, re.IGNORECASE)
    if m: return m.group(1)
    return None

def extract_goal_name(msg: str) -> Optional[str]:
    """Extract goal name from message."""
    m = re.search(r"['\"]([^'\"]+)['\"]", msg)
    if m: return m.group(1)
    for kw in ["goal", "target"]:
        m = re.search(rf'{kw}\s+(?:called\s+|named\s+)?(\w[\w\s]*?)(?:\s+(?:to|by|deadline)|$)', msg, re.IGNORECASE)
        if m: return m.group(1).strip()
    return None


# ─── Finance Kernel ───────────────────────────────────────────────────────────

# Global Shared State
_EXACT_CACHE: dict[str, str] = {}
_FUZZY_CACHE: list[tuple[str, str]] = []
_CHAT_HISTORY: dict[str, list] = {}  # session_id -> [{role, content}]

class FinanceKernel:
    def __init__(self, user_id: str, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.user_chroma_path = ScopeManager(level="user").resolve(user_id=self.user_id)
        self.session_chroma_path = ScopeManager(level="session").resolve(
            user_id=self.user_id, session_id=self.session_id
        )
        self.kernel = None
        self._booted = False
        self._action_tools: dict = {}
        self.cache_col = None

    async def boot(self):
        if self._booted:
            return

        import asyncio
        def _sync_boot():
            # Compulsorily instantiate a NEW kernel using Ollama provider to use the local fast model
            config = KernelConfig(
                provider="ollama",
                model="phi3:mini",
                embedding_model="BAAI/bge-small-en-v1.5",
                require_consensus=False,
                verbose=False
            )
            k = SumoKernel(config)
            
            # Point to this user's memory (runs in thread to avoid ONNX deadlocks)
            k._memory = MemoryManager(
                chroma_path=self.session_chroma_path,
                embedding_provider=k.config.embedding_provider,
                user_id=self.user_id,
                session_id=self.session_id
            )
            # Run initialization synchronously inside this thread
            import asyncio
            try: 
                asyncio.run(k.boot())
                asyncio.run(k._memory.initialize())
            except RuntimeError: pass  # Already running event loop fallback
            
            return k

        self.kernel = await asyncio.to_thread(_sync_boot)

        # Initialize Semantic Cache (lightweight)
        def _init_cache():
            cache_client = chromadb.PersistentClient(path=os.path.join(self.user_chroma_path, "cache"))
            return cache_client.get_or_create_collection(
                name="semantic_cache", 
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
        try:
            self.cache_col = await asyncio.to_thread(_init_cache)
        except: pass

        # Map intent actions to tool runners
        self._action_tools = {
            "update_merchant": self._handle_update_merchant,
            "refuse_suggestion": self._handle_refuse,
            "update_goal": self._handle_update_goal,
            "add_bill": self._handle_add_bill,
            "savings_schedule": self._handle_savings,
            "pause_goal": self._handle_pause_goal,
            "unpause_goal": self._handle_unpause_goal,
            "mock_transaction": self._handle_mock_txn,
        }

        # Initialize chat history for this session
        if self.session_id not in _CHAT_HISTORY:
            _CHAT_HISTORY[self.session_id] = []

        self._booted = True

    # ─── Action Handlers ──────────────────────────────────────────────────

    async def _handle_update_merchant(self, msg: str) -> str:
        merchant = extract_merchant(msg) or "Unknown"
        rank = "Essential" if "essential" in msg.lower() else "Avoidable"
        cat_tool = tools.CategorizationTool()
        res = await cat_tool.run(self.user_id, merchant)
        # Infer category from existing data or default
        cat = "Misc"
        override = db.get_merchant_override(self.user_id, merchant)
        if override: cat = override["category"]
        tool = tools.UpdateMerchantRankTool(kernel_ref=self)
        result = await tool.run(self.user_id, merchant, rank, cat)
        return f"✅ Done! {merchant} is now marked as **{rank}** ({cat})."

    async def _handle_refuse(self, msg: str) -> str:
        merchant = extract_merchant(msg) or "this item"
        tool = tools.RefusedSuggestionTool(kernel_ref=self)
        await tool.run(self.user_id, "reduce_spending", merchant)
        return f"✅ Got it! I'll never recommend reducing spending at **{merchant}** again."

    async def _handle_update_goal(self, msg: str) -> str:
        goal_name = extract_goal_name(msg) or "Unknown"
        end_date = None
        m = re.search(r'(?:to|by)\s+(\w+\s+\d{4}|\d{4}-\d{2}-\d{2})', msg, re.IGNORECASE)
        if m: end_date = m.group(1)
        amount = extract_amount(msg)
        tool = tools.UpdateGoalTool(kernel_ref=self)
        result = await tool.run(self.user_id, goal_name, target_amount=amount, end_date=end_date)
        return f"✅ {result.output}"

    async def _handle_create_goal(self, msg: str) -> str:
        goal_name = extract_goal_name(msg) or "New Goal"
        amount = extract_amount(msg)
        
        # If amount or date is completely missing from message, we should ask user.
        # But we don't have direct LLM context here, so we will return a system prompt
        # that the kernel will pass to the LLM to format as a question.
        if not amount:
            return "SYSTEM_INSTRUCTION: Inform the user that you can help them create the goal, but you need to know the target amount and the deadline. Ask them for this information."
            
        # Basic date extraction heuristic
        end_date = None
        import re
        from datetime import datetime, timedelta
        # Match YYYY-MM-DD
        date_m = re.search(r"(\d{4}-\d{2}-\d{2})", msg)
        if date_m:
            end_date = date_m.group(1)
        else:
            # Check for year
            year_m = re.search(r"\b(202\d)\b", msg)
            if year_m:
                end_date = f"{year_m.group(1)}-12-31"
        
        if not end_date:
            return "SYSTEM_INSTRUCTION: Inform the user that you need a deadline or target date to create the goal."
            
        tool = tools.CreateGoalTool(kernel_ref=self)
        result = await tool.run(self.user_id, goal_name, amount, end_date)
        return f"✅ {result.output}"

    async def _handle_add_bill(self, msg: str) -> str:
        merchant = extract_merchant(msg) or "Unknown"
        amount = extract_amount(msg) or 0
        tool = tools.AddFixedBillTool(kernel_ref=self)
        result = await tool.run(self.user_id, merchant, amount)
        return f"✅ {result.output}"

    async def _handle_savings(self, msg: str) -> str:
        amount = extract_amount(msg) or 5000
        goals = db.get_goals(self.user_id)
        active = [g for g in goals if g["status"] == "active"]
        if active:
            g = active[0]
            tool = tools.SavingsScheduleTool(kernel_ref=self)
            new_target = g["target_amount"] + amount
            result = await tool.run(self.user_id, g["name"], new_target, g["start_date"], g["end_date"])
            return f"✅ Added ₹{amount:,.0f} to your '{g['name']}' goal target. {result.output}"
        return f"No active goals found. Create a goal first to set a savings target."

    async def _handle_pause_goal(self, msg: str) -> str:
        goal_name = extract_goal_name(msg) or ""
        tool = tools.PauseGoalTool(kernel_ref=self)
        result = await tool.run(self.user_id, goal_name, action="pause")
        return f"✅ {result.output}" if result.success else f"❌ {result.output}"

    async def _handle_unpause_goal(self, msg: str) -> str:
        goal_name = extract_goal_name(msg) or ""
        tool = tools.PauseGoalTool(kernel_ref=self)
        result = await tool.run(self.user_id, goal_name, action="unpause")
        return f"✅ {result.output}" if result.success else f"❌ {result.output}"

    async def _handle_mock_txn(self, msg: str) -> str:
        amount = extract_amount(msg) or 100
        merchant = extract_merchant(msg) or "Unknown"
        tool = tools.MockTransactionTool(kernel_ref=self)
        result = await tool.run(self.user_id, merchant, amount)
        return f"✅ {result.output}"

    async def _handle_discover_schemes(self, msg: str) -> str:
        tool = tools.SchemeDiscoveryTool(kernel_ref=self)
        result = await tool.run(self.user_id)
        # Give the LLM the text output so it can summarize/present it nicely
        return f"SYSTEM_INSTRUCTION: Inform the user about these schemes clearly: {result.output}"

    # ─── VDB Ingestion ────────────────────────────────────────────────────

    async def _ingest_text(self, user_id: str, text: str, metadata: dict):
        """Stub — ingestion is handled by seed_schemes.py directly now."""
        pass

    async def _recall_user_context(self, user_id: str, query: str, top_k: int = 20):
        """Stub — context is built from live SQLite data now."""
        return []

    async def sync_full_context_to_vdb(self):
        """Ingest ALL user data into VDB for comprehensive RAG retrieval."""
        if not self._booted:
            await self.boot()
        user = db.get_user(self.user_id)
        if not user: return

        chunks = []
        # User profile
        income = db.get_dynamic_income(self.user_id)
        chunks.append(f"User: {user['name']}, Income ₹{income:,.0f}/month, Risk: {user.get('risk_profile','moderate')}, Balance: ₹{user['current_balance']:,.0f}")

        # Goals
        for g in db.get_goals(self.user_id):
            chunks.append(f"Goal '{g['name']}': ₹{g['current_savings']:,.0f}/₹{g['target_amount']:,.0f}, {g['start_date']}→{g['end_date']}, Status: {g['status']}")

        # Fixed bills
        bills = db.get_fixed_bills(self.user_id)
        if bills:
            bill_str = ", ".join(f"{b['merchant']} ₹{b['amount']:,.0f}" for b in bills)
            chunks.append(f"Fixed bills: {bill_str}")

        # Budgets
        budgets = db.get_budgets(self.user_id)
        if budgets:
            budget_str = ", ".join(f"{b['category']} ₹{b['monthly_limit']:,.0f}/mo" for b in budgets)
            chunks.append(f"Budgets: {budget_str}")

        # Monthly summary
        month = datetime.now().strftime("%Y-%m")
        summary = db.get_monthly_summary(self.user_id, month)
        chunks.append(f"Summary {month}: Income ₹{summary['total_income']:,.0f}, Spent ₹{summary['total_expense']:,.0f}, Saved ₹{summary['savings']:,.0f} ({summary['savings_rate']*100:.1f}%)")

        # Recent transactions (last 30)
        txns = db.get_transactions(self.user_id, limit=30)
        for t in txns:
            chunks.append(f"{t['type'].upper()} ₹{t['amount']:,.0f} at {t['merchant']} ({t['category']}) on {t['timestamp'][:10]}")

        for chunk in chunks:
            try:
                await self._ingest_text(self.user_id, chunk, {"type": "context_sync"})
            except Exception:
                pass

    # ─── Live Context Builder ─────────────────────────────────────────────

    def _build_live_context(self) -> str:
        """Build structured text from live SQLite data."""
        user = db.get_user(self.user_id)
        if not user:
            return "No user data found."

        parts = []
        income = db.get_dynamic_income(self.user_id)
        parts.append(f"USER: {user['name']} | Balance: ₹{user['current_balance']:,.0f} | Avg Income: ₹{income:,.0f}/month")

        # Goals
        goals = db.get_goals(self.user_id)
        if goals:
            goal_lines = []
            for g in goals:
                pct = (g["current_savings"] / g["target_amount"] * 100) if g["target_amount"] > 0 else 0
                goal_lines.append(f"  - {g['name']}: ₹{g['current_savings']:,.0f}/₹{g['target_amount']:,.0f} ({pct:.0f}%) [{g['status']}] ends {g['end_date']}")
            parts.append("GOALS:\n" + "\n".join(goal_lines))

        # Bills
        bills = db.get_fixed_bills(self.user_id)
        if bills:
            parts.append("FIXED BILLS: " + ", ".join(f"{b['merchant']} ₹{b['amount']:,.0f}" for b in bills))

        # Recent transactions
        txns = db.get_transactions(self.user_id, limit=15)
        if txns:
            txn_lines = [f"  - {t['type']} ₹{t['amount']:,.0f} at {t['merchant']} ({t['category']}) {t['timestamp'][:10]}" for t in txns]
            parts.append("RECENT TRANSACTIONS:\n" + "\n".join(txn_lines))

        # Spending by category this month
        month = datetime.now().strftime("%Y-%m")
        cats = db.get_spending_by_category(self.user_id, month)
        if cats:
            total = sum(c["total"] for c in cats) or 1
            cat_lines = [f"  - {c['category']}: ₹{c['total']:,.0f} ({c['total']/total*100:.0f}%)" for c in cats]
            parts.append(f"SPENDING THIS MONTH ({month}):\n" + "\n".join(cat_lines))

        # Budget status
        breaches = db.check_budget_breach(self.user_id)
        if breaches:
            parts.append("BUDGET ALERTS: " + ", ".join(
                f"{'⚠️OVER' if b['breached'] else '⚡NEAR'} {b['category']} {b['percentage']:.0f}%" for b in breaches))

        return "\n\n".join(parts)

    # ─── Main Chat Entry Point ────────────────────────────────────────────

    async def chat(self, message: str) -> str:
        """3-phase chat: detect_intent → context_assembly → direct_ollama."""
        if not self._booted:
            await self.boot()

        # Phase 1: Intent detection — is this an ACTION or a QUERY?
        action = detect_intent(message)
        if action and action in self._action_tools:
            handler = self._action_tools[action]
            try:
                result = await handler(message)
                _CHAT_HISTORY.setdefault(self.session_id, []).append({"role": "user", "content": message})
                _CHAT_HISTORY[self.session_id].append({"role": "assistant", "content": result})
                return result
            except Exception as e:
                pass  # Fall through to query path



        # Phase 2: Parallel Context Assembly (Speed Boost)
        import asyncio
        
        # Build live context (instant)
        live_context = await asyncio.to_thread(self._build_live_context)
        
        # Retrieve RAG context (semantic matching, executed in thread)
        def _get_rag_context():
            try:
                # We use asyncio.run because we are inside a background thread
                import asyncio
                rag_result = asyncio.run(self.kernel._rag.retrieve(message))
                return rag_result.context if rag_result and rag_result.chunks else ""
            except:
                return ""
                
        rag_context = await asyncio.to_thread(_get_rag_context)

        # Get recent conversation history
        history_msgs = _CHAT_HISTORY.get(self.session_id, [])[-6:]
        history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history_msgs)

        # Phase 3: LLM synthesis
        user_prompt = f"""=== LIVE FINANCIAL DATA ===
{live_context}

=== RELEVANT HISTORY & PATTERNS ===
{rag_context if rag_context else 'No historical patterns stored yet.'}

=== CONVERSATION ===
{history if history else '(new conversation)'}

USER: {message}"""

        print(f"🧠 Loading model and generating response for query: '{message}' ... (This may take 1-3 minutes on CPU)", flush=True)

        def _run_llm():
            import requests
            messages = [{"role": "system", "content": FINANCE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "phi3:mini",
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
                timeout=600
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        try:
            response = await asyncio.to_thread(_run_llm)
        except Exception as e:
            response = f"I encountered an issue processing your request ({type(e).__name__}: {str(e)}). Here is your live data: {live_context[:300]}"

        # Save to conversation history
        _CHAT_HISTORY.setdefault(self.session_id, []).append({"role": "user", "content": message})
        _CHAT_HISTORY[self.session_id].append({"role": "assistant", "content": response})
        if len(_CHAT_HISTORY[self.session_id]) > 20:
            _CHAT_HISTORY[self.session_id] = _CHAT_HISTORY[self.session_id][-10:]

            
        return response
