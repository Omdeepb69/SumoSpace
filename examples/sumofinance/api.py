"""SumoFinance API — 28 endpoints across Auth, Dashboard, Insights, Recommendations, Simulation, Chat, Goals, Nudges, Transactions."""
from fastapi import FastAPI, HTTPException, Body, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import db
from finance_kernel import FinanceKernel
import tools
import auth
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="SumoFinance API", description="AI-Powered Financial Decision Engine with Bank-Grade 2FA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for the tunnel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── Auth Dependency ──────────────────────────────────────────────────────────

async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict | None:
    if not credentials: return None
    payload = auth.verify_jwt(credentials.credentials)
    if not payload: raise HTTPException(401, "Invalid or expired token")
    user = auth.get_auth_user(payload["sub"])
    if not user: raise HTTPException(401, "User not found")
    return user

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str
    email: str
    initial_balance: float = 0.0

class GoalCreate(BaseModel):
    name: str
    target_amount: float
    start_date: str
    end_date: str

class GoalUpdate(BaseModel):
    target_amount: Optional[float] = None
    end_date: Optional[str] = None

class TransactionCreate(BaseModel):
    merchant: str
    amount: float
    category: str = ""
    type: str = "expense"

class ChatRequest(BaseModel):
    message: str

class MerchantUpdate(BaseModel):
    merchant: str
    rank: str
    category: str

class PlanSelect(BaseModel):
    plan: str  # easy | moderate | aggressive

class SimulationRequest(BaseModel):
    category: str
    target_reduction: float = 20
    time_horizon_months: int = 6

class InteractiveSimRequest(BaseModel):
    category: str
    slider_value: float = 20

class WhatIfRequest(BaseModel):
    recommendation_id: str

class RecommendationContext(BaseModel):
    recommendation_id: str

# ─── Kernel Cache ─────────────────────────────────────────────────────────────

_kernel_cache: dict[str, FinanceKernel] = {}

async def get_kernel(user_id: str, session_id: str = "default") -> FinanceKernel:
    key = f"{user_id}:{session_id}"
    if key not in _kernel_cache:
        fk = FinanceKernel(user_id=user_id, session_id=session_id)
        await fk.boot()
        _kernel_cache[key] = fk
    return _kernel_cache[key]

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS (4)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=auth.RegisterResponse, tags=["Auth"])
async def register(req: auth.RegisterRequest):
    try: return auth.register_user(req)
    except ValueError as e: raise HTTPException(400, str(e))

@app.post("/auth/verify-2fa", response_model=auth.TokenResponse, tags=["Auth"])
async def verify_2fa(req: auth.VerifyOTPRequest):
    try: return auth.verify_otp(req.user_id, req.otp_code)
    except ValueError as e: raise HTTPException(401, str(e))

@app.post("/auth/login", response_model=auth.LoginResponse, tags=["Auth"])
async def login(req: auth.LoginRequest):
    try: return auth.login_step1(req.email, req.password)
    except ValueError as e: raise HTTPException(401, str(e))

@app.post("/auth/login/2fa", response_model=auth.TokenResponse, tags=["Auth"])
async def login_2fa(req: auth.VerifyOTPRequest):
    try: return auth.verify_otp(req.user_id, req.otp_code)
    except ValueError as e: raise HTTPException(401, str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS (kept for backward compat)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/users", tags=["Legacy"])
async def create_user(user: UserCreate):
    user_id = db.create_user(user.name, user.email, user.initial_balance)
    return {"user_id": user_id, "status": "created"}

@app.patch("/merchant/{user_id}", tags=["Legacy"])
async def update_merchant(user_id: str, req: MerchantUpdate):
    db.set_merchant_override(user_id, req.merchant, req.rank, req.category)
    return {"status": "updated"}

@app.get("/history/{user_id}", tags=["Legacy"])
async def get_history(user_id: str):
    return {"fixed_bills": db.get_fixed_bills(user_id), "preferences": db.get_preferences(user_id)}

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINTS (3)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/summary/{user_id}", tags=["Dashboard"])
async def dashboard_summary(user_id: str):
    user = db.get_user(user_id)
    if not user: raise HTTPException(404, "User not found")
    tool = tools.SpendingSummaryTool()
    res = await tool.run(user_id=user_id)
    s2s = tools.SafeToSpendTool()
    s2s_res = await s2s.run(user_id=user_id)
    return {
        "user": {"name": user["name"], "balance": user["current_balance"],
                 "income": db.get_dynamic_income(user_id)},
        "spending": res.metadata,
        "safe_to_spend": s2s_res.metadata.get("safe_to_spend", 0) if s2s_res.success else 0,
    }

# Keep old dashboard for backward compat
@app.get("/dashboard/{user_id}", tags=["Dashboard"])
async def get_dashboard(user_id: str):
    user = db.get_user(user_id)
    if not user: raise HTTPException(404, "User not found")
    s2s = tools.SafeToSpendTool()
    s2s_res = await s2s.run(user_id=user_id)
    return {
        "user": user, "goals": db.get_goals(user_id),
        "safe_to_spend_analysis": s2s_res.output,
        "safe_to_spend_amount": s2s_res.metadata.get("safe_to_spend", 0) if s2s_res.success else 0,
    }

@app.get("/dashboard/spending-trend/{user_id}", tags=["Dashboard"])
async def spending_trend(user_id: str, view: str = Query("daily", enum=["daily", "weekly", "monthly"]),
                         period: str = Query("")):
    data = db.get_spending_trend(user_id, view, period or None)
    return {"view": view, "data": data}

@app.get("/dashboard/alerts/{user_id}", tags=["Dashboard"])
async def dashboard_alerts(user_id: str):
    tool = tools.AlertGeneratorTool()
    res = await tool.run(user_id=user_id)
    return res.metadata

@app.get("/dashboard/daily-list/{user_id}", tags=["Dashboard"])
async def daily_list(user_id: str):
    """Return daily spend aggregation for the Spending Calendar heatmap."""
    txns = db.get_transactions(user_id, limit=1000)
    daily: dict[str, dict] = {}
    for t in txns:
        if t["type"] != "expense":
            continue
        try:
            day = t["timestamp"][:10]  # "YYYY-MM-DD"
        except Exception:
            continue
        if day not in daily:
            daily[day] = {"amount": 0, "cats": {}}
        daily[day]["amount"] += t["amount"]
        cat = t.get("category", "Other")
        daily[day]["cats"][cat] = daily[day]["cats"].get(cat, 0) + t["amount"]

    days = []
    for date_str in sorted(daily.keys()):
        entry = daily[date_str]
        top_cat = max(entry["cats"], key=entry["cats"].get) if entry["cats"] else ""
        days.append({"date": date_str, "amount": round(entry["amount"], 2), "top_category": top_cat})
    return {"status": "success", "days": days}

# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHTS ENDPOINTS (5)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/insights/category-breakdown/{user_id}", tags=["Insights"])
async def category_breakdown(user_id: str, period: str = Query("")):
    tool = tools.CategoryBreakdownTool()
    res = await tool.run(user_id=user_id, period=period)
    return res.metadata

@app.get("/insights/ai-observations/{user_id}", tags=["Insights"])
async def ai_observations(user_id: str):
    """Hybrid: instant heuristics + optional LLM narrative."""
    observations = []
    now = datetime.now()
    month = now.strftime("%Y-%m")

    # 1. Weekend spike detection
    txns = db.get_transactions(user_id, limit=500)
    weekend_spend, weekday_spend, we_count, wd_count = 0, 0, 0, 0
    for t in txns:
        if t["type"] != "expense": continue
        try:
            dt = datetime.fromisoformat(t["timestamp"])
            if dt.weekday() >= 5:
                weekend_spend += t["amount"]; we_count += 1
            else:
                weekday_spend += t["amount"]; wd_count += 1
        except: pass
    we_avg = weekend_spend / max(we_count, 1)
    wd_avg = weekday_spend / max(wd_count, 1)
    if we_avg > wd_avg * 1.5:
        observations.append({
            "type": "pattern", "title": "Weekend spending spike",
            "description": f"You spend {we_avg/wd_avg:.1f}x more on weekends (₹{we_avg:,.0f} vs ₹{wd_avg:,.0f} avg)",
            "severity": "warning", "affected_category": "all",
            "suggested_action": "Set a weekend spending cap"
        })

    # 2. Budget breaches
    breaches = db.check_budget_breach(user_id, month)
    for b in breaches:
        observations.append({
            "type": "warning", "title": f"Budget {'exceeded' if b['breached'] else 'nearly exceeded'}: {b['category']}",
            "description": f"₹{b['spent']:,.0f} of ₹{b['limit']:,.0f} ({b['percentage']:.0f}%)",
            "severity": "critical" if b["breached"] else "warning",
            "affected_category": b["category"],
            "suggested_action": f"Reduce {b['category']} spending by ₹{max(0, b['spent']-b['limit']):,.0f}"
        })

    # 3. Anomaly detection
    anomalies = db.get_anomalies(user_id)
    for a in anomalies[:3]:
        observations.append({
            "type": "anomaly", "title": f"Unusual: ₹{a['amount']:,.0f} at {a['merchant']}",
            "description": f"{a['z_score']}σ above your avg ₹{a['category_mean']:,.0f} for {a['category']}",
            "severity": "info", "affected_category": a["category"],
            "suggested_action": "Review if this was intentional"
        })

    # 4. Category creep (month-over-month)
    comparison = tools.SpendingComparisonTool()
    comp_res = await comparison.run(user_id=user_id)
    if comp_res.success:
        for d in comp_res.metadata.get("differences", []):
            if d["change_pct"] > 15:
                observations.append({
                    "type": "trend", "title": f"{d['category']} spending up {d['change_pct']:+.0f}%",
                    "description": f"₹{d['current']:,.0f} this month vs ₹{d['previous']:,.0f} last month",
                    "severity": "warning", "affected_category": d["category"],
                    "suggested_action": f"Review your {d['category']} spending"
                })

    # 5. Merchant loyalty
    top_merchants = db.get_top_merchants(user_id, limit=20)
    for m in top_merchants:
        if m["frequency"] >= 15:
            observations.append({
                "type": "pattern", "title": f"Frequent: {m['merchant']} ({m['frequency']}x)",
                "description": f"₹{m['total_spent']:,.0f} total at {m['merchant']}, avg ₹{m['avg_amount']:,.0f}",
                "severity": "info", "affected_category": m["category"],
                "suggested_action": "Consider if all visits are necessary"
            })

    # 6. Income-to-spend ratio
    ratios = db.get_income_expense_ratio(user_id, months=3)
    improving = all(ratios[i]["ratio"] >= ratios[i+1]["ratio"] for i in range(len(ratios)-1)) if len(ratios) >= 2 else False
    if ratios and not improving:
        observations.append({
            "type": "trend", "title": "Savings rate declining",
            "description": f"Income-to-expense ratio trending down over last 3 months",
            "severity": "warning", "affected_category": "all",
            "suggested_action": "Review your spending categories for reduction opportunities"
        })

    return {"observations": observations, "count": len(observations)}

@app.get("/insights/comparison/{user_id}", tags=["Insights"])
async def spending_comparison(user_id: str, current: str = Query(""), previous: str = Query("")):
    tool = tools.SpendingComparisonTool()
    res = await tool.run(user_id=user_id, current_period=current, previous_period=previous)
    return res.metadata

@app.get("/insights/merchant-analysis/{user_id}", tags=["Insights"])
async def merchant_analysis(user_id: str, period: str = Query("")):
    tool = tools.MerchantAnalysisTool()
    res = await tool.run(user_id=user_id, period=period)
    return res.metadata

@app.get("/insights/anomalies/{user_id}", tags=["Insights"])
async def anomalies(user_id: str, period: str = Query("")):
    tool = tools.AnomalyDetectorTool()
    res = await tool.run(user_id=user_id, period=period)
    return res.metadata

# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS ENDPOINTS (4)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/recommendations/list/{user_id}", tags=["Recommendations"])
async def recommendations_list(user_id: str, plan: str = Query("moderate", enum=["easy", "moderate", "aggressive"])):
    tool = tools.RecommendationEngineTool()
    res = await tool.run(user_id=user_id, plan=plan)
    return res.metadata

@app.post("/recommendations/select-plan/{user_id}", tags=["Recommendations"])
async def select_plan(user_id: str, req: PlanSelect):
    tool = tools.RecommendationEngineTool()
    res = await tool.run(user_id=user_id, plan=req.plan)
    return res.metadata

@app.get("/recommendations/strategy-summary/{user_id}", tags=["Recommendations"])
async def strategy_summary(user_id: str):
    tool = tools.RecommendationEngineTool()
    res = await tool.run(user_id=user_id, plan="moderate")
    recs = res.metadata.get("recommendations", [])
    strategy = f"Focus on reducing high-frequency variable categories first. Total potential: ₹{res.metadata.get('total_savings', 0):,.0f}/year."
    return {"total_possible_saving": res.metadata.get("total_savings", 0),
            "strategy": strategy, "recommendation_count": len(recs)}

@app.get("/recommendations/history/{user_id}", tags=["Recommendations"])
async def recommendation_history(user_id: str, status: str = Query("")):
    return {"recommendations": db.get_recommendations(user_id, status or None)}

@app.get("/recommendations/schemes/{user_id}", tags=["Recommendations"])
async def recommend_schemes(user_id: str):
    tool = tools.SchemeDiscoveryTool()
    res = await tool.run(user_id=user_id)
    if res.success:
        return {"schemes": res.metadata.get("schemes", []), "surplus": res.metadata.get("surplus", 0), "goal": res.metadata.get("goal", ""), "message": res.output}
    return {"schemes": [], "surplus": 0, "goal": "", "message": res.output}

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENDPOINTS (3)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/simulation/run/{user_id}", tags=["Simulation"])
async def run_simulation(user_id: str, req: SimulationRequest):
    tool = tools.SimulationEngineTool()
    res = await tool.run(user_id=user_id, category=req.category,
                         target_reduction=req.target_reduction,
                         time_horizon_months=req.time_horizon_months)
    if not res.success: raise HTTPException(400, res.output)
    return res.metadata

@app.post("/simulation/what-if-not-followed/{user_id}", tags=["Simulation"])
async def what_if_not_followed(user_id: str, req: WhatIfRequest):
    recs = db.get_recommendations(user_id)
    rec = next((r for r in recs if r["rec_id"] == req.recommendation_id), None)
    if not rec:
        # Generate from current data
        return {"impact": {"yearly_loss": 0, "message": "Recommendation not found."}}
    return {"impact": {"yearly_loss": rec["impact_yearly"],
                       "message": f"Ignoring '{rec['title']}' costs you ₹{rec['impact_yearly']:,.0f}/year."}}

@app.get("/simulation/prediction/{user_id}", tags=["Simulation"])
async def get_prediction(user_id: str):
    tool = tools.BehavioralPredictorTool()
    res = await tool.run(user_id=user_id)
    if res.success:
        return res.metadata
    return {}
@app.post("/simulation/interactive/{user_id}", tags=["Simulation"])
async def interactive_sim(user_id: str, req: InteractiveSimRequest):
    tool = tools.InteractiveSimulatorTool()
    res = await tool.run(user_id=user_id, category=req.category, slider_value=req.slider_value)
    if not res.success: raise HTTPException(400, res.output)
    return res.metadata

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ENDPOINTS (2)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/chat/{user_id}/{session_id}", tags=["Chat"])
async def chat(user_id: str, session_id: str, req: ChatRequest):
    print(f"✅ Received chat request from {user_id}. Query: '{req.message}'", flush=True)
    try:
        fk = await get_kernel(user_id, session_id)
        response = await fk.chat(req.message)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)[:200]}"}

@app.post("/chat/{user_id}/reset-session", tags=["Chat"])
async def reset_session(user_id: str):
    keys_to_remove = [k for k in _kernel_cache if k.startswith(f"{user_id}:")]
    for k in keys_to_remove:
        del _kernel_cache[k]
    return {"status": "session_reset", "message": "Chat session cleared."}

# ═══════════════════════════════════════════════════════════════════════════════
# GOALS ENDPOINTS (4)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/goals/{user_id}", tags=["Goals"])
async def create_goal(user_id: str, goal: GoalCreate):
    goal_id = db.create_goal(user_id, goal.name, goal.target_amount, goal.start_date, goal.end_date)
    return {"goal_id": goal_id, "status": "created"}

@app.patch("/goals/{user_id}/{goal_id}", tags=["Goals"])
async def update_goal(user_id: str, goal_id: str, req: GoalUpdate):
    ok = db.update_goal(user_id, goal_id, target_amount=req.target_amount, end_date=req.end_date)
    if not ok: raise HTTPException(404, "Goal not found")
    return {"status": "updated"}

@app.post("/goals/{user_id}/{goal_id}/pause", tags=["Goals"])
async def pause_goal(user_id: str, goal_id: str, action: str = Query("pause", enum=["pause", "unpause"])):
    if action == "pause":
        ok = db.pause_goal(user_id, goal_id)
    else:
        ok = db.unpause_goal(user_id, goal_id)
    if not ok: raise HTTPException(404, "Goal not found or already in that state")
    return {"status": action + "d"}

@app.get("/progress/{user_id}", tags=["Goals"])
async def get_progress(user_id: str):
    user = db.get_user(user_id)
    if not user: raise HTTPException(404, "User not found")
    goals = db.get_goals(user_id)
    month = datetime.now().strftime("%Y-%m")
    summary = db.get_monthly_summary(user_id, month)
    target = db.get_dynamic_income(user_id) * 0.2  # Default 20% savings target
    return {"current_saving": summary["savings"], "target": target,
            "progress": round(summary["savings"] / target * 100, 1) if target > 0 else 0,
            "goals": [{"name": g["name"], "progress": round(g["current_savings"]/g["target_amount"]*100, 1) if g["target_amount"] > 0 else 0,
                       "status": g["status"]} for g in goals]}

# ═══════════════════════════════════════════════════════════════════════════════
# NUDGES ENDPOINTS (3)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/nudges/{user_id}", tags=["Nudges"])
async def get_nudges(user_id: str, unread_only: bool = Query(False)):
    return {"nudges": db.get_nudges(user_id, unread_only)}

@app.get("/nudges/{user_id}/unread-count", tags=["Nudges"])
async def unread_nudge_count(user_id: str):
    return {"count": db.get_unread_nudge_count(user_id)}

@app.post("/nudges/{user_id}/{nudge_id}/read", tags=["Nudges"])
async def mark_nudge_read(user_id: str, nudge_id: str):
    ok = db.mark_nudge_read(user_id, nudge_id)
    if not ok: raise HTTPException(404, "Nudge not found")
    return {"status": "read"}

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTIONS ENDPOINTS (2)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/transaction/{user_id}", tags=["Transactions"])
async def process_transaction(user_id: str, txn: TransactionCreate):
    category = txn.category
    if not category:
        m = tools.MERCHANT_MAP.get(txn.merchant, {"category": "Misc"})
        category = m["category"]
    txn_id = db.add_transaction(user_id, txn.amount, category, txn.merchant, txn.type)
    return {"txn_id": txn_id, "status": "processed", "category": category}

@app.get("/transactions/{user_id}", tags=["Transactions"])
async def get_transactions(user_id: str, start: str = Query(""), end: str = Query(""),
                           category: str = Query(""), limit: int = Query(100)):
    txns = db.get_transactions(user_id, start or None, end or None, category or None, limit)
    return {"transactions": txns, "count": len(txns)}

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/features/{user_id}", tags=["Features"])
async def get_features(user_id: str):
    return {"features": db.get_feature_flags(user_id)}

@app.patch("/features/{user_id}/{flag_name}", tags=["Features"])
async def toggle_feature(user_id: str, flag_name: str, enabled: bool = Query(True)):
    db.set_feature_flag(user_id, flag_name, enabled)
    return {"status": "updated", "flag": flag_name, "enabled": enabled}

# ═══════════════════════════════════════════════════════════════════════════════
# SEED DATA ENDPOINT (for demo)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/seed/{user_id}", tags=["Admin"])
async def seed_data(user_id: str):
    from seed_data import seed_user_data
    seed_user_data(user_id)
    return {"status": "seeded", "message": "150+ transactions, bills, budgets, goals created."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
