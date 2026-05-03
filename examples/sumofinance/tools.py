"""SumoFinance — Deterministic Tools (no LLM, pure math)."""
from datetime import datetime, timedelta
import math, statistics, json, re
from typing import Optional, Any
from sumospace.tools import BaseTool, ToolResult
import db

MERCHANT_MAP = {
    "Amazon": {"category": "Shopping", "rank": "Avoidable"},
    "Swiggy": {"category": "Food", "rank": "Avoidable"},
    "Zomato": {"category": "Food", "rank": "Avoidable"},
    "Netflix": {"category": "Entertainment", "rank": "Avoidable"},
    "Uber": {"category": "Transport", "rank": "Avoidable"},
    "Salary": {"category": "Income", "rank": "Essential"},
    "Gym": {"category": "Health", "rank": "Essential"},
    "DMart": {"category": "Groceries", "rank": "Essential"},
    "Starbucks": {"category": "Food", "rank": "Avoidable"},
}

# ─── Existing Tools (kept) ────────────────────────────────────────────────────

class CategorizationTool(BaseTool):
    name = "categorize_merchant"
    description = "Gets category and rank of a merchant."
    async def run(self, user_id: str, merchant: str, **_) -> ToolResult:
        override = db.get_merchant_override(user_id, merchant)
        if override:
            cat, rank = override["category"], override["rank"]
        else:
            d = MERCHANT_MAP.get(merchant, {"category": "Misc", "rank": "Avoidable"})
            cat, rank = d["category"], d["rank"]
        return ToolResult(tool=self.name, success=True,
                          output=f"'{merchant}': {cat} ({rank})")

class SafeToSpendTool(BaseTool):
    name = "calculate_safe_to_spend"
    description = "Calculates safe-to-spend amount."
    async def run(self, user_id: str, **_) -> ToolResult:
        user = db.get_user(user_id)
        if not user:
            return ToolResult(tool=self.name, success=False, output="User not found.")
        balance = user["current_balance"]
        bills = db.get_fixed_bills(user_id)
        fixed = sum(b["amount"] for b in bills)
        goals = db.get_goals(user_id)
        goal_contrib = sum(max((g["target_amount"] - g["current_savings"]) / 12, 0)
                          for g in goals if g["status"] == "active")
        s2s = balance - (fixed + 500 + goal_contrib)
        output = f"Balance: ₹{balance:,.0f} | Bills: ₹{fixed:,.0f} | Goals: ₹{goal_contrib:,.0f} | Safe-to-Spend: ₹{s2s:,.0f}"
        return ToolResult(tool=self.name, success=True, output=output,
                          metadata={"safe_to_spend": s2s})

class GoalProgressTool(BaseTool):
    name = "goal_progress"
    description = "Calculates goal progress percentage."
    async def run(self, user_id: str, goal_name: str = "", **_) -> ToolResult:
        goals = db.get_goals(user_id)
        if goal_name:
            goals = [g for g in goals if g["name"].lower() == goal_name.lower()]
        if not goals:
            return ToolResult(tool=self.name, success=False, output=f"Goal '{goal_name}' not found.")
        lines = []
        for g in goals:
            pct = (g["current_savings"] / g["target_amount"] * 100) if g["target_amount"] > 0 else 0
            lines.append(f"'{g['name']}': ₹{g['current_savings']:,.0f}/₹{g['target_amount']:,.0f} ({pct:.1f}%) [{g['status']}]")
        return ToolResult(tool=self.name, success=True, output="\n".join(lines))

class SavingsScheduleTool(BaseTool):
    name = "savings_schedule"
    description = "Computes daily/weekly/monthly savings for a goal."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, goal_name: str, target_amount: float,
                  start_date: str, end_date: str, **_) -> ToolResult:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        days = (end - start).days
        if days <= 0:
            return ToolResult(tool=self.name, success=False, output="End date must be after start.")
        daily, weekly, monthly = target_amount/days, target_amount/days*7, target_amount/days*30
        txt = f"Goal '{goal_name}': Save ₹{daily:.0f}/day, ₹{weekly:.0f}/week, ₹{monthly:.0f}/month ({start_date} → {end_date})"
        if self.kernel:
            try: await self.kernel._ingest_text(user_id, txt, {"type": "goal_schedule", "goal": goal_name})
            except: pass
        return ToolResult(tool=self.name, success=True, output=txt)

class MockTransactionTool(BaseTool):
    name = "mock_transaction"
    description = "Creates a transaction, updates balance, ingests to VDB."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, merchant: str, amount: float,
                  txn_type: str = "expense", category: str = "", **_) -> ToolResult:
        if not category:
            d = MERCHANT_MAP.get(merchant, {"category": "Misc"})
            category = d["category"]
        txn_id = db.add_transaction(user_id, amount, category, merchant, txn_type)
        txt = f"{txn_type.upper()} ₹{amount:,.0f} at {merchant} ({category}) on {datetime.now().strftime('%Y-%m-%d')}"
        if self.kernel:
            try: await self.kernel._ingest_text(user_id, txt, {"type": "transaction", "merchant": merchant})
            except: pass
        return ToolResult(tool=self.name, success=True, output=f"Processed: {txt}")

class UpdateMerchantRankTool(BaseTool):
    name = "update_merchant_rank"
    description = "Updates merchant rank and category."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, merchant: str, rank: str, category: str, **_) -> ToolResult:
        db.set_merchant_override(user_id, merchant, rank, category)
        txt = f"'{merchant}' → {rank}/{category}"
        if self.kernel:
            try: await self.kernel._ingest_text(user_id, txt, {"type": "preference", "merchant": merchant})
            except: pass
        return ToolResult(tool=self.name, success=True, output=txt)

class RefusedSuggestionTool(BaseTool):
    name = "record_refused_suggestion"
    description = "Records a refused suggestion so AI never recommends it again."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, suggestion_type: str, merchant: str, **_) -> ToolResult:
        db.add_preference(user_id, "refused_suggestion", f"Refused {suggestion_type} for {merchant}")
        rule = f"User refused '{suggestion_type}' for {merchant} on {datetime.now().strftime('%Y-%m-%d')}. Never recommend again."
        if self.kernel:
            try: await self.kernel._ingest_text(user_id, rule, {"type": "preference", "constraint": "do_not_recommend"})
            except: pass
        return ToolResult(tool=self.name, success=True, output=rule)

class PatternDetectorTool(BaseTool):
    name = "detect_patterns"
    description = "Queries VDB for spending patterns."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, query: str = "spending patterns", **_) -> ToolResult:
        if not self.kernel:
            return ToolResult(tool=self.name, success=False, output="Kernel missing.")
        chunks = await self.kernel._recall_user_context(user_id, query, top_k=20)
        output = f"Retrieved {len(chunks)} chunks:\n" + "\n".join(c.text for c in chunks)
        return ToolResult(tool=self.name, success=True, output=output)

class ScenarioSimulatorTool(BaseTool):
    name = "simulate_scenario"
    description = "Simulates how a spend affects a goal."
    async def run(self, user_id: str, goal_name: str, spend_amount: float, **_) -> ToolResult:
        goals = db.get_goals(user_id)
        goal = next((g for g in goals if g["name"].lower() == goal_name.lower()), None)
        if not goal:
            return ToolResult(tool=self.name, success=False, output=f"Goal '{goal_name}' not found.")
        new = goal["current_savings"] - spend_amount
        output = f"Spending ₹{spend_amount:,.0f} drops '{goal_name}' savings to ₹{max(new,0):,.0f}"
        return ToolResult(tool=self.name, success=True, output=output)

# ─── NEW Phase 2 Tools ────────────────────────────────────────────────────────

class SpendingSummaryTool(BaseTool):
    name = "spending_summary"
    description = "Aggregates spending totals for week/month/year."
    async def run(self, user_id: str, **_) -> ToolResult:
        now = datetime.now()
        month = now.strftime("%Y-%m")
        week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        year = now.strftime("%Y")
        m_summary = db.get_monthly_summary(user_id, month)
        w_txns = db.get_transactions(user_id, start_date=week_start)
        week_total = sum(t["amount"] for t in w_txns if t["type"] == "expense")
        y_summary = db.get_monthly_summary(user_id, year)
        data = {
            "month": {"total_spent": m_summary["total_expense"], "savings": m_summary["savings"],
                      "savings_rate": m_summary["savings_rate"]},
            "week": {"total_spent": round(week_total, 2)},
            "year": {"total_spent": y_summary["total_expense"]},
        }
        output = f"Month: ₹{m_summary['total_expense']:,.0f} spent, ₹{m_summary['savings']:,.0f} saved | Week: ₹{week_total:,.0f} | Year: ₹{y_summary['total_expense']:,.0f}"
        return ToolResult(tool=self.name, success=True, output=output, metadata=data)

class CategoryBreakdownTool(BaseTool):
    name = "category_breakdown"
    description = "Per-category spending with percentages."
    async def run(self, user_id: str, period: str = "", **_) -> ToolResult:
        period = period or datetime.now().strftime("%Y-%m")
        cats = db.get_spending_by_category(user_id, period)
        total = sum(c["total"] for c in cats) or 1
        result = [{"name": c["category"], "amount": round(c["total"], 2),
                    "percentage": round(c["total"]/total*100, 1), "count": c["count"]} for c in cats]
        lines = [f"{r['name']}: ₹{r['amount']:,.0f} ({r['percentage']}%)" for r in result]
        return ToolResult(tool=self.name, success=True,
                          output=f"Breakdown for {period}:\n" + "\n".join(lines),
                          metadata={"period": period, "categories": result})

class SpendingComparisonTool(BaseTool):
    name = "spending_comparison"
    description = "Compares spending between two periods."
    async def run(self, user_id: str, current_period: str = "", previous_period: str = "", **_) -> ToolResult:
        now = datetime.now()
        current_period = current_period or now.strftime("%Y-%m")
        prev_dt = now.replace(day=1) - timedelta(days=1)
        previous_period = previous_period or prev_dt.strftime("%Y-%m")
        cur = db.get_spending_by_category(user_id, current_period)
        prev = db.get_spending_by_category(user_id, previous_period)
        prev_map = {c["category"]: c["total"] for c in prev}
        diffs = []
        for c in cur:
            old = prev_map.get(c["category"], 0)
            change = ((c["total"] - old) / old * 100) if old > 0 else 100
            behavior = "worsened" if change > 5 else "improved" if change < -5 else "stable"
            diffs.append({"category": c["category"], "current": round(c["total"],2),
                          "previous": round(old,2), "change_pct": round(change,1), "behavior": behavior})
        lines = [f"{d['category']}: {d['change_pct']:+.1f}% ({d['behavior']})" for d in diffs]
        return ToolResult(tool=self.name, success=True,
                          output=f"{current_period} vs {previous_period}:\n" + "\n".join(lines),
                          metadata={"current": current_period, "previous": previous_period, "differences": diffs})

class AlertGeneratorTool(BaseTool):
    name = "generate_alerts"
    description = "Detects overspending, budget breaches, anomalies."
    async def run(self, user_id: str, **_) -> ToolResult:
        alerts = []
        breaches = db.check_budget_breach(user_id)
        for b in breaches:
            sev = "critical" if b["breached"] else "warning"
            alerts.append({"type": sev, "title": f"{'Over' if b['breached'] else 'Near'} budget: {b['category']}",
                           "description": f"Spent ₹{b['spent']:,.0f} of ₹{b['limit']:,.0f} ({b['percentage']:.0f}%)",
                           "category": b["category"]})
        anomalies = db.get_anomalies(user_id)
        for a in anomalies[:3]:
            alerts.append({"type": "warning", "title": f"Unusual: ₹{a['amount']:,.0f} at {a['merchant']}",
                           "description": f"This is {a['z_score']}σ above your avg ₹{a['category_mean']:,.0f} for {a['category']}",
                           "category": a["category"]})
        return ToolResult(tool=self.name, success=True,
                          output=f"{len(alerts)} alerts generated",
                          metadata={"alerts": alerts})

class MerchantAnalysisTool(BaseTool):
    name = "merchant_analysis"
    description = "Top merchants by spend, frequency, avg transaction."
    async def run(self, user_id: str, period: str = "", limit: int = 10, **_) -> ToolResult:
        merchants = db.get_top_merchants(user_id, limit, period or None)
        result = [{"merchant": m["merchant"], "category": m["category"],
                    "total_spent": round(m["total_spent"],2), "frequency": m["frequency"],
                    "avg_amount": round(m["avg_amount"],2)} for m in merchants]
        lines = [f"{r['merchant']}: ₹{r['total_spent']:,.0f} ({r['frequency']}x, avg ₹{r['avg_amount']:,.0f})" for r in result]
        return ToolResult(tool=self.name, success=True,
                          output="Top merchants:\n" + "\n".join(lines),
                          metadata={"merchants": result})

class AnomalyDetectorTool(BaseTool):
    name = "detect_anomalies"
    description = "Flags transactions >2σ above category mean."
    async def run(self, user_id: str, period: str = "", **_) -> ToolResult:
        anomalies = db.get_anomalies(user_id, period or None)
        result = [{"merchant": a["merchant"], "amount": a["amount"], "category": a["category"],
                    "z_score": a["z_score"], "mean": a["category_mean"]} for a in anomalies]
        lines = [f"₹{a['amount']:,.0f} at {a['merchant']} ({a['z_score']}σ above avg ₹{a['mean']:,.0f})" for a in result]
        return ToolResult(tool=self.name, success=True,
                          output=f"{len(result)} anomalies:\n" + "\n".join(lines),
                          metadata={"anomalies": result})

class RecommendationEngineTool(BaseTool):
    name = "generate_recommendations"
    description = "Generates savings plans: easy/moderate/aggressive."
    async def run(self, user_id: str, plan: str = "moderate", **_) -> ToolResult:
        cats = db.get_spending_by_category(user_id)
        reducible_cats = []
        all_cats_db = {c["name"]: c for c in db.get_categories()}
        for c in cats:
            meta = all_cats_db.get(c["category"], {"is_reducible": 1})
            if meta.get("is_reducible", 1):
                reducible_cats.append(c)
        multiplier = {"easy": 0.10, "moderate": 0.20, "aggressive": 0.35}.get(plan, 0.20)
        recs = []
        total_savings = 0
        for c in reducible_cats:
            monthly_save = c["total"] * multiplier
            yearly_save = monthly_save * 12
            difficulty = "easy" if multiplier <= 0.1 else "medium" if multiplier <= 0.2 else "hard"
            prob = max(0.3, 0.85 - multiplier)
            recs.append({"title": f"Reduce {c['category']} by {int(multiplier*100)}%",
                         "description": f"Cut ₹{monthly_save:,.0f}/month from {c['category']}",
                         "category": c["category"], "impact_monthly": round(monthly_save,2),
                         "impact_yearly": round(yearly_save,2), "difficulty": difficulty,
                         "success_probability": round(prob,2)})
            total_savings += yearly_save
        return ToolResult(tool=self.name, success=True,
                          output=f"{plan} plan: {len(recs)} recommendations, ₹{total_savings:,.0f}/year potential savings",
                          metadata={"plan": plan, "recommendations": recs, "total_savings": round(total_savings,2)})

class SimulationEngineTool(BaseTool):
    name = "run_simulation"
    description = "Full simulation with habit model, behaviour drift, stochastic range."
    async def run(self, user_id: str, category: str, target_reduction: float = 20,
                  time_horizon_months: int = 6, **_) -> ToolResult:
        cats = db.get_spending_by_category(user_id)
        cat_data = next((c for c in cats if c["category"].lower() == category.lower()), None)
        
        # If no data, provide a default 0-spend simulation instead of failing
        if not cat_data:
            cat_data = {"category": category, "total": 0.0, "count": 0}
            
        monthly_avg = cat_data["total"]
        # Gather last 3 months for stochastic range
        now = datetime.now()
        monthly_spends = []
        for i in range(3):
            dt = now - timedelta(days=30*i)
            m = dt.strftime("%Y-%m")
            s = db.get_spending_by_category(user_id, m)
            cat_s = next((x for x in s if x["category"].lower() == category.lower()), None)
            if cat_s:
                monthly_spends.append(cat_s["total"])
        if not monthly_spends:
            monthly_spends = [monthly_avg]
        mean_spend = statistics.mean(monthly_spends)
        std_spend = statistics.stdev(monthly_spends) if len(monthly_spends) > 1 else mean_spend * 0.15
        # Budget breach count for resistance
        budgets = db.get_budgets(user_id)
        budget_for_cat = next((b for b in budgets if b["category"].lower() == category.lower()), None)
        overspend_count = sum(1 for ms in monthly_spends if budget_for_cat and ms > budget_for_cat["monthly_limit"])
        resistance = "high" if overspend_count >= 2 else "medium" if overspend_count >= 1 else "low"
        adoption = "gradual" if resistance in ("high", "medium") else "immediate"
        # Monthly projection
        projections = []
        total_saved = 0
        for month in range(1, time_horizon_months + 1):
            effective_target = min(target_reduction, target_reduction * (month / 3))
            decay = max(1.0 - 0.1 * (month - 1), 0.3)
            effective = effective_target * decay
            saved = mean_spend * (effective / 100)
            total_saved += saved
            projections.append({"month": month, "reduction_pct": round(effective, 1), "savings": round(saved, 0)})
        drift = [{"month": 1, "discipline": "high"},
                 {"month": 3, "discipline": "medium" if resistance != "low" else "high"},
                 {"month": 6, "discipline": "low" if resistance == "high" else "medium"}]
        result = {
            "category": category, "monthly_avg": round(mean_spend, 0),
            "monthly_projection": projections,
            "habit_model": {"category": category, "frequency": cat_data["count"],
                            "resistance": resistance, "expected_adoption": adoption},
            "behavior_drift": drift,
            "stochastic_range": {"expected": round(total_saved, 0),
                                 "variance": round(std_spend * time_horizon_months, 0)},
            "total_projected_savings": round(total_saved, 0),
        }
        return ToolResult(tool=self.name, success=True,
                          output=f"Simulation: {category} -{target_reduction}% over {time_horizon_months}mo → save ₹{total_saved:,.0f}",
                          metadata=result)

class InteractiveSimulatorTool(BaseTool):
    name = "interactive_simulation"
    description = "Instant slider-based what-if calculation."
    async def run(self, user_id: str, category: str, slider_value: float = 20, **_) -> ToolResult:
        cats = db.get_spending_by_category(user_id)
        # If no data, provide a default 0-spend result instead of failing
        if not cat_data:
            cat_data = {"category": category, "total": 0.0, "count": 0}
            
        monthly = cat_data["total"] * (slider_value / 100)
        yearly = monthly * 12
        prob = max(0.3, 0.85 - slider_value/100)
        return ToolResult(tool=self.name, success=True,
                          output=f"Cut {category} by {slider_value}% → save ₹{monthly:,.0f}/mo, ₹{yearly:,.0f}/yr",
                          metadata={"monthly_saving": round(monthly,2), "yearly_saving": round(yearly,2),
                                    "success_probability": round(prob,2)})

class NudgeGeneratorTool(BaseTool):
    name = "generate_nudges"
    description = "Creates budget-aware nudge alerts."
    async def run(self, user_id: str, **_) -> ToolResult:
        breaches = db.check_budget_breach(user_id)
        nudges = []
        for b in breaches:
            msg = f"You've spent {b['percentage']:.0f}% of your {b['category']} budget (₹{b['spent']:,.0f}/₹{b['limit']:,.0f})"
            if b["breached"]:
                msg = f"⚠️ OVER BUDGET: {msg}"
            nid = db.add_nudge(user_id, msg)
            nudges.append({"nudge_id": nid, "message": msg})
        return ToolResult(tool=self.name, success=True,
                          output=f"{len(nudges)} nudges created",
                          metadata={"nudges": nudges})

class UpdateGoalTool(BaseTool):
    name = "update_goal"
    description = "Updates a goal's target or end date."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, goal_name: str, target_amount: float = None,
                  end_date: str = None, **_) -> ToolResult:
        goals = db.get_goals(user_id)
        goal = next((g for g in goals if g["name"].lower() == goal_name.lower()), None)
        if not goal:
            return ToolResult(tool=self.name, success=False, output=f"Goal '{goal_name}' not found.")
        db.update_goal(user_id, goal["goal_id"], target_amount=target_amount, end_date=end_date)
        parts = []
        if target_amount: parts.append(f"target→₹{target_amount:,.0f}")
        if end_date: parts.append(f"deadline→{end_date}")
        txt = f"Goal '{goal_name}' updated: {', '.join(parts)}"
        if self.kernel:
            try: await self.kernel._ingest_text(user_id, txt, {"type": "goal_update", "goal": goal_name})
            except: pass
        return ToolResult(tool=self.name, success=True, output=txt)

class PauseGoalTool(BaseTool):
    name = "pause_goal"
    description = "Pauses or unpauses a goal."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, goal_name: str, action: str = "pause", **_) -> ToolResult:
        goals = db.get_goals(user_id)
        goal = next((g for g in goals if g["name"].lower() == goal_name.lower()), None)
        if not goal:
            return ToolResult(tool=self.name, success=False, output=f"Goal '{goal_name}' not found.")
        if action == "pause":
            ok = db.pause_goal(user_id, goal["goal_id"])
        else:
            ok = db.unpause_goal(user_id, goal["goal_id"])
        txt = f"Goal '{goal_name}' {'paused' if action == 'pause' else 'resumed'}"
        if self.kernel and ok:
            try: await self.kernel._ingest_text(user_id, txt, {"type": "goal_status", "goal": goal_name})
            except: pass
        return ToolResult(tool=self.name, success=True if ok else False, output=txt)

class AddFixedBillTool(BaseTool):
    name = "add_fixed_bill"
    description = "Adds a recurring fixed bill."
    def __init__(self, kernel_ref=None): self.kernel = kernel_ref
    async def run(self, user_id: str, merchant: str, amount: float,
                  frequency: str = "monthly", **_) -> ToolResult:
        db.add_fixed_bill(user_id, merchant, amount, frequency)
        txt = f"Fixed bill: {merchant} ₹{amount:,.0f} ({frequency})"
        if self.kernel:
            try: await self.kernel._ingest_text(user_id, txt, {"type": "fixed_bill", "merchant": merchant})
            except: pass
        return ToolResult(tool=self.name, success=True, output=txt)

class CreateGoalTool(BaseTool):
    name = "create_goal"
    description = "Creates a new financial goal with a target amount and an end date."
    input_schema = {
        "user_id": {"type": "string", "description": "The user's ID"},
        "name": {"type": "string", "description": "The name or purpose of the goal"},
        "target_amount": {"type": "number", "description": "The target amount to save"},
        "end_date": {"type": "string", "description": "The end date in YYYY-MM-DD format. If unspecified, assume 1 year from now."}
    }

    def _execute(self, user_id: str, name: str, target_amount: float, end_date: str = None) -> ToolResult:
        if not end_date:
            end_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
            
        start_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            goal_id = db.create_goal(user_id, name, target_amount, start_date, end_date)
            
            # Sync to VDB
            from finance_kernel import _sync_goals
            _sync_goals(user_id)
            
            return ToolResult(success=True, output=f"Created goal '{name}' for ₹{target_amount} to be reached by {end_date}.")
        except Exception as e:
            return ToolResult(success=False, output=f"Failed to create goal: {str(e)}")

class SchemeDiscoveryTool(BaseTool):
    name = "discover_schemes"
    description = "Discovers investment schemes based on the user's surplus, goals, and income."
    
    def __init__(self, kernel_ref=None):
        self.kernel = kernel_ref
        
    async def run(self, user_id: str, **_) -> ToolResult:
        # 1. Get surplus
        s2s_tool = SafeToSpendTool()
        s2s_res = await s2s_tool.run(user_id=user_id)
        surplus = s2s_res.metadata.get("safe_to_spend", 0) if s2s_res.success else 0
        
        if surplus <= 0:
            return ToolResult(tool=self.name, success=False, output="No surplus available. Focus on building an emergency fund first.")
            
        # 2. Get goals to find timeline
        goals = db.get_goals(user_id)
        active_goals = [g for g in goals if g["status"] == "active"]
        
        # Default timeline if no goals: 60 months (long-term)
        timeline_months = 60
        goal_name = "Long-term Wealth Creation"
        
        if active_goals:
            # Get the nearest goal
            def months_until(end_date_str):
                try:
                    ed = datetime.strptime(end_date_str, "%Y-%m-%d")
                    return max(1, (ed.year - datetime.now().year) * 12 + ed.month - datetime.now().month)
                except:
                    return 12
            active_goals.sort(key=lambda g: months_until(g["end_date"]))
            nearest_goal = active_goals[0]
            timeline_months = months_until(nearest_goal["end_date"])
            goal_name = nearest_goal["name"]
            
        # 3. Get Income
        income = db.get_dynamic_income(user_id)
        
        # 4. Fetch all schemes and apply hard filters
        all_schemes = db.get_schemes()
        filtered = []
        for s in all_schemes:
            if s["minimum_deposit"] > surplus:
                continue
            if s["lock_in_months"] > timeline_months:
                continue
            
            # Subsidies targeting
            if s["category"] == "Insurance" and "government" in s["tags"].lower() and income > 30000:
                continue # Skip low-income govt schemes if user earns well
                
            filtered.append(s)
            
        if not filtered:
            return ToolResult(tool=self.name, success=True, metadata={"schemes": [], "goal": goal_name}, output="No matching schemes found for your timeline.")
            
        # Sort by returns
        filtered.sort(key=lambda x: x["returns_percentage"], reverse=True)
        top_schemes = filtered[:3]
        
        out_txt = f"Since you have ₹{surplus:,.0f} surplus and are saving for '{goal_name}' (in {timeline_months} months):\n"
        for idx, s in enumerate(top_schemes):
            out_txt += f"{idx+1}. {s['name']} ({s['category']}) - {s['returns_percentage']}% returns. {s['fine_print']}\n"
            
        return ToolResult(tool=self.name, success=True, metadata={
            "surplus": surplus,
            "goal": goal_name,
            "timeline": timeline_months,
            "schemes": top_schemes
        }, output=out_txt)

class BehavioralPredictorTool(BaseTool):
    name = "predict_behavior"
    description = "Analyzes transaction patterns to predict end of month spending and issue warnings."
    
    async def run(self, user_id: str, **_) -> ToolResult:
        # 1. Fetch last 90 days of expense transactions
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        txns = db.get_transactions(user_id, start_date=start_date)
        expenses = [t for t in txns if t["type"] == "expense" and t["category"] != "Bills"]
        
        if not expenses:
            return ToolResult(tool=self.name, success=False, output="Not enough data to predict patterns.")
            
        # 2. Analyze Weekday vs Weekend spending
        import pandas as pd
        df = pd.DataFrame(expenses)
        df['date'] = pd.to_datetime(df['timestamp'])
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        # Daily averages
        daily_spend = df.groupby(['date', 'is_weekend'])['amount'].sum().reset_index()
        avg_weekend = daily_spend[daily_spend['is_weekend'] == True]['amount'].mean()
        avg_weekday = daily_spend[daily_spend['is_weekend'] == False]['amount'].mean()
        
        avg_weekend = avg_weekend if pd.notna(avg_weekend) else 0
        avg_weekday = avg_weekday if pd.notna(avg_weekday) else 0
        
        # 3. Predict end of month based on remaining days
        now = datetime.now()
        import calendar
        _, last_day = calendar.monthrange(now.year, now.month)
        remaining_days = last_day - now.day
        
        rem_weekends = sum(1 for d in range(now.day + 1, last_day + 1) if datetime(now.year, now.month, d).weekday() >= 5)
        rem_weekdays = remaining_days - rem_weekends
        
        projected_spend = (rem_weekdays * avg_weekday) + (rem_weekends * avg_weekend)
        
        # Current month spend
        month_str = now.strftime("%Y-%m")
        month_summary = db.get_monthly_summary(user_id, month_str)
        current_spend = month_summary.get("total_spent", 0)
        
        total_projected = current_spend + projected_spend
        
        # 4. Generate Warnings
        warnings = []
        if avg_weekend > (avg_weekday * 1.5) and avg_weekend > 0:
            warnings.append(f"🚨 You spend {avg_weekend/max(avg_weekday, 1):.1f}x more on weekends (₹{avg_weekend:,.0f}/day vs ₹{avg_weekday:,.0f}/day).")
            warnings.append(f"💡 Suggestion: Cap your upcoming weekend spending to ₹{avg_weekday * 1.2:,.0f} to increase your surplus.")
            
        if avg_weekend < (avg_weekday * 0.5) and avg_weekday > 0:
            warnings.append(f"🚨 You spend significantly more on weekdays (₹{avg_weekday:,.0f}/day). Check your daily work expenses like food and transit.")
            
        txt = (
            f"**Current Month Spend**: ₹{current_spend:,.0f}\n"
            f"**Projected End of Month**: ₹{total_projected:,.0f}\n\n"
            f"**Behavioral Patterns**:\n" + "\n".join(warnings)
        )
        
        return ToolResult(tool=self.name, success=True, metadata={
            "avg_weekend": avg_weekend,
            "avg_weekday": avg_weekday,
            "projected_eom": total_projected,
            "current_spend": current_spend,
            "warnings": warnings
        }, output=txt)
