import sqlite3
import uuid
import statistics
from datetime import datetime, timedelta
from typing import Optional
from sumospace.scope import ScopeManager

DB_PATH = "sumofinance.db"


# ─── Schema ──────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            created_at TEXT,
            current_balance REAL,
            monthly_income REAL DEFAULT 0,
            risk_profile TEXT DEFAULT 'moderate'
        );
        CREATE TABLE IF NOT EXISTS goals (
            goal_id TEXT PRIMARY KEY,
            user_id TEXT,
            name TEXT,
            target_amount REAL,
            current_savings REAL,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE IF NOT EXISTS fixed_bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            merchant TEXT,
            amount REAL,
            frequency TEXT DEFAULT 'monthly'
        );
        CREATE TABLE IF NOT EXISTS merchant_overrides (
            user_id TEXT,
            merchant TEXT,
            rank TEXT,
            category TEXT,
            updated_at TEXT,
            PRIMARY KEY (user_id, merchant)
        );
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            preference_key TEXT,
            preference_value TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS transactions (
            txn_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            merchant TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT DEFAULT 'expense'
        );
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY,
            type TEXT DEFAULT 'variable',
            is_reducible INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS budgets (
            user_id TEXT,
            category TEXT,
            monthly_limit REAL,
            PRIMARY KEY (user_id, category)
        );
        CREATE TABLE IF NOT EXISTS nudges (
            nudge_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            read INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS feature_flags (
            user_id TEXT NOT NULL,
            flag_name TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            PRIMARY KEY (user_id, flag_name)
        );
        CREATE TABLE IF NOT EXISTS recommendation_history (
            rec_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            description TEXT,
            category TEXT,
            impact_yearly REAL,
            difficulty TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'suggested',
            created_at TEXT,
            resolved_at TEXT
        );
        CREATE TABLE IF NOT EXISTS investment_schemes (
            scheme_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            returns_percentage REAL,
            lock_in_months INTEGER,
            minimum_deposit REAL,
            risk_level TEXT,
            source_url TEXT,
            fine_print TEXT,
            tags TEXT
        );
    ''')
    conn.commit()
    conn.close()


def seed_categories():
    """Pre-populate default spending categories."""
    defaults = [
        ("Food", "variable", 1),
        ("Transport", "variable", 1),
        ("Shopping", "variable", 1),
        ("Entertainment", "variable", 1),
        ("Groceries", "variable", 1),
        ("Health", "variable", 0),
        ("Bills", "fixed", 0),
        ("Rent", "fixed", 0),
        ("Education", "fixed", 0),
        ("Software", "variable", 1),
        ("Income", "fixed", 0),
        ("Savings", "fixed", 0),
        ("Misc", "variable", 1),
    ]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for name, typ, reducible in defaults:
        c.execute("INSERT OR IGNORE INTO categories (name, type, is_reducible) VALUES (?, ?, ?)",
                  (name, typ, reducible))
    conn.commit()
    conn.close()


# ─── User CRUD ────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, initial_balance: float = 0.0,
                monthly_income: float = 0.0, risk_profile: str = "moderate") -> str:
    user_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (user_id, name, email, created_at, current_balance, monthly_income, risk_profile) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, name, email, datetime.now().isoformat(), initial_balance, monthly_income, risk_profile)
    )
    conn.commit()
    conn.close()
    scope = ScopeManager(level="user")
    scope.resolve(user_id=user_id)
    return user_id


def get_user(user_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def update_balance(user_id: str, new_balance: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET current_balance = ? WHERE user_id = ?", (new_balance, user_id))
    conn.commit()
    conn.close()

def get_dynamic_income(user_id: str, months_to_average: int = 3) -> float:
    """Calculates average monthly income dynamically from 'income' transactions over the last N months."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all income transactions for the last N months
    now = datetime.now()
    cutoff_date = (now - timedelta(days=30 * months_to_average)).isoformat()
    
    c.execute(
        "SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = 'income' AND timestamp >= ?",
        (user_id, cutoff_date)
    )
    total_income = c.fetchone()[0]
    conn.close()
    
    if total_income is None:
        return 0.0
        
    return total_income / months_to_average


# ─── Goals ────────────────────────────────────────────────────────────────────

def create_goal(user_id: str, name: str, target_amount: float,
                start_date: str, end_date: str) -> str:
    goal_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO goals (goal_id, user_id, name, target_amount, current_savings, start_date, end_date, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (goal_id, user_id, name, target_amount, 0.0, start_date, end_date, 'active')
    )
    conn.commit()
    conn.close()
    return goal_id


def get_goals(user_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM goals WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_goal(user_id: str, goal_id: str,
                target_amount: Optional[float] = None,
                end_date: Optional[str] = None,
                current_savings: Optional[float] = None) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    updates, params = [], []
    if target_amount is not None:
        updates.append("target_amount = ?"); params.append(target_amount)
    if end_date is not None:
        updates.append("end_date = ?"); params.append(end_date)
    if current_savings is not None:
        updates.append("current_savings = ?"); params.append(current_savings)
    if not updates:
        conn.close()
        return False
    params.extend([goal_id, user_id])
    c.execute(f"UPDATE goals SET {', '.join(updates)} WHERE goal_id = ? AND user_id = ?", params)
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


def pause_goal(user_id: str, goal_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE goals SET status = 'paused' WHERE goal_id = ? AND user_id = ? AND status = 'active'",
              (goal_id, user_id))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


def unpause_goal(user_id: str, goal_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE goals SET status = 'active' WHERE goal_id = ? AND user_id = ? AND status = 'paused'",
              (goal_id, user_id))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


# ─── Fixed Bills ──────────────────────────────────────────────────────────────

def add_fixed_bill(user_id: str, merchant: str, amount: float,
                   frequency: str = "monthly") -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO fixed_bills (user_id, merchant, amount, frequency) VALUES (?, ?, ?, ?)",
        (user_id, merchant, amount, frequency)
    )
    conn.commit()
    bill_id = c.lastrowid
    conn.close()
    return bill_id


def get_fixed_bills(user_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM fixed_bills WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ─── Merchant Overrides ──────────────────────────────────────────────────────

def set_merchant_override(user_id: str, merchant: str, rank: str, category: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO merchant_overrides (user_id, merchant, rank, category, updated_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, merchant, rank, category, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_merchant_override(user_id: str, merchant: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM merchant_overrides WHERE user_id = ? AND merchant = ?", (user_id, merchant))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


# ─── Preferences ──────────────────────────────────────────────────────────────

def add_preference(user_id: str, key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO user_preferences (user_id, preference_key, preference_value, created_at) VALUES (?, ?, ?, ?)",
        (user_id, key, value, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_preferences(user_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_user_data(user_id: str):
    """Deletes all user transactions, goals, budgets, and nudges before seeding."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM goals WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM budgets WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM nudges WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM fixed_bills WHERE user_id = ?", (user_id,))
    c.execute("UPDATE users SET current_balance = 0 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# ─── Transactions ─────────────────────────────────────────────────────────────

def add_transaction(user_id: str, amount: float, category: str, merchant: str,
                    txn_type: str = "expense", timestamp: Optional[str] = None) -> str:
    txn_id = str(uuid.uuid4())
    ts = timestamp or datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO transactions (txn_id, user_id, amount, category, merchant, timestamp, type) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (txn_id, user_id, amount, category, merchant, ts, txn_type)
    )
    # Update balance
    if txn_type == "income":
        c.execute("UPDATE users SET current_balance = current_balance + ? WHERE user_id = ?", (amount, user_id))
    else:
        c.execute("UPDATE users SET current_balance = current_balance - ? WHERE user_id = ?", (amount, user_id))
    conn.commit()
    conn.close()
    return txn_id


def get_transactions(user_id: str, start_date: Optional[str] = None,
                     end_date: Optional[str] = None, category: Optional[str] = None,
                     limit: int = 100) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    query = "SELECT * FROM transactions WHERE user_id = ?"
    params: list = [user_id]
    if start_date:
        query += " AND timestamp >= ?"; params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"; params.append(end_date)
    if category:
        query += " AND category = ?"; params.append(category)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_spending_by_category(user_id: str, period: Optional[str] = None) -> list[dict]:
    """Returns [{category, total, count}] for a given month (YYYY-MM) or all time."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    query = "SELECT category, SUM(amount) as total, COUNT(*) as count FROM transactions WHERE user_id = ? AND type = 'expense'"
    params: list = [user_id]
    if period:
        query += " AND timestamp LIKE ?"
        params.append(f"{period}%")
    query += " GROUP BY category ORDER BY total DESC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_spending_trend(user_id: str, view: str = "daily",
                       period: Optional[str] = None) -> list[dict]:
    """Returns time-series spending data. view: daily | weekly | monthly."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if view == "daily":
        date_expr = "DATE(timestamp)"
    elif view == "weekly":
        date_expr = "strftime('%Y-W%W', timestamp)"
    else:  # monthly
        date_expr = "strftime('%Y-%m', timestamp)"

    query = f"SELECT {date_expr} as date, SUM(amount) as amount FROM transactions WHERE user_id = ? AND type = 'expense'"
    params: list = [user_id]
    if period:
        query += " AND timestamp LIKE ?"
        params.append(f"{period}%")
    query += f" GROUP BY {date_expr} ORDER BY date"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_top_merchants(user_id: str, limit: int = 10,
                      period: Optional[str] = None) -> list[dict]:
    """Returns top merchants by total spend."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    query = """SELECT merchant, category, SUM(amount) as total_spent,
               COUNT(*) as frequency, AVG(amount) as avg_amount
               FROM transactions WHERE user_id = ? AND type = 'expense'"""
    params: list = [user_id]
    if period:
        query += " AND timestamp LIKE ?"
        params.append(f"{period}%")
    query += " GROUP BY merchant ORDER BY total_spent DESC LIMIT ?"
    params.append(limit)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_monthly_summary(user_id: str, month: str) -> dict:
    """Get total income, expense, savings for a YYYY-MM month."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = 'expense' AND timestamp LIKE ?",
              (user_id, f"{month}%"))
    total_expense = c.fetchone()[0] or 0
    c.execute("SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = 'income' AND timestamp LIKE ?",
              (user_id, f"{month}%"))
    total_income = c.fetchone()[0] or 0
    conn.close()
    savings = total_income - total_expense
    savings_rate = (savings / total_income) if total_income > 0 else 0
    return {"month": month, "total_income": total_income, "total_expense": total_expense,
            "savings": savings, "savings_rate": round(savings_rate, 4)}


def get_anomalies(user_id: str, period: Optional[str] = None,
                  threshold_sigma: float = 2.0) -> list[dict]:
    """Find transactions that are > threshold_sigma standard deviations above the mean for their category."""
    cats = get_spending_by_category(user_id, period)
    anomalies = []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    for cat_row in cats:
        cat = cat_row["category"]
        query = "SELECT * FROM transactions WHERE user_id = ? AND category = ? AND type = 'expense'"
        params: list = [user_id, cat]
        if period:
            query += " AND timestamp LIKE ?"
            params.append(f"{period}%")
        c.execute(query, params)
        txns = [dict(r) for r in c.fetchall()]
        if len(txns) < 3:
            continue
        amounts = [t["amount"] for t in txns]
        mean = statistics.mean(amounts)
        std = statistics.stdev(amounts)
        if std == 0:
            continue
        for t in txns:
            z = (t["amount"] - mean) / std
            if z > threshold_sigma:
                t["z_score"] = round(z, 2)
                t["category_mean"] = round(mean, 2)
                t["category_std"] = round(std, 2)
                anomalies.append(t)
    conn.close()
    return anomalies


def get_income_expense_ratio(user_id: str, months: int = 3) -> list[dict]:
    """Returns income/expense ratio for the last N months."""
    ratios = []
    now = datetime.now()
    for i in range(months):
        dt = now - timedelta(days=30 * i)
        month_str = dt.strftime("%Y-%m")
        summary = get_monthly_summary(user_id, month_str)
        ratio = (summary["total_income"] / summary["total_expense"]) if summary["total_expense"] > 0 else 0
        ratios.append({"month": month_str, "ratio": round(ratio, 2), **summary})
    return ratios


# ─── Budgets ──────────────────────────────────────────────────────────────────

def set_budget(user_id: str, category: str, monthly_limit: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO budgets (user_id, category, monthly_limit) VALUES (?, ?, ?)",
              (user_id, category, monthly_limit))
    conn.commit()
    conn.close()


def get_budgets(user_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM budgets WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def check_budget_breach(user_id: str, month: Optional[str] = None) -> list[dict]:
    """Check which categories have breached their budget."""
    month = month or datetime.now().strftime("%Y-%m")
    budgets = get_budgets(user_id)
    spending = get_spending_by_category(user_id, month)
    spend_map = {s["category"]: s["total"] for s in spending}
    breaches = []
    for b in budgets:
        spent = spend_map.get(b["category"], 0)
        pct = (spent / b["monthly_limit"] * 100) if b["monthly_limit"] > 0 else 0
        if pct >= 80:
            breaches.append({
                "category": b["category"],
                "limit": b["monthly_limit"],
                "spent": spent,
                "percentage": round(pct, 1),
                "breached": pct >= 100,
            })
    return breaches


# ─── Nudges ───────────────────────────────────────────────────────────────────

def add_nudge(user_id: str, message: str) -> str:
    nudge_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO nudges (nudge_id, user_id, message, created_at) VALUES (?, ?, ?, ?)",
              (nudge_id, user_id, message, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return nudge_id


def get_nudges(user_id: str, unread_only: bool = False) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    query = "SELECT * FROM nudges WHERE user_id = ?"
    params: list = [user_id]
    if unread_only:
        query += " AND read = 0"
    query += " ORDER BY created_at DESC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_nudge_read(user_id: str, nudge_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE nudges SET read = 1 WHERE nudge_id = ? AND user_id = ?", (nudge_id, user_id))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


def get_unread_nudge_count(user_id: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM nudges WHERE user_id = ? AND read = 0", (user_id,))
    count = c.fetchone()[0]
    conn.close()
    return count


# ─── Feature Flags ────────────────────────────────────────────────────────────

DEFAULT_FLAGS = {
    "time_simulation": True,
    "habit_modeling": True,
    "event_modeling": True,
    "stochastic_simulation": True,
    "behavior_drift": True,
    "ai_observations": True,
    "nudges": True,
}


def init_feature_flags(user_id: str):
    """Initialize default feature flags for a new user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for flag, enabled in DEFAULT_FLAGS.items():
        c.execute("INSERT OR IGNORE INTO feature_flags (user_id, flag_name, enabled) VALUES (?, ?, ?)",
                  (user_id, flag, int(enabled)))
    conn.commit()
    conn.close()


def set_feature_flag(user_id: str, flag_name: str, enabled: bool):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO feature_flags (user_id, flag_name, enabled) VALUES (?, ?, ?)",
              (user_id, flag_name, int(enabled)))
    conn.commit()
    conn.close()


def get_feature_flags(user_id: str) -> dict[str, bool]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM feature_flags WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    flags = {**DEFAULT_FLAGS}
    for row in rows:
        flags[row["flag_name"]] = bool(row["enabled"])
    return flags


def is_feature_enabled(user_id: str, flag_name: str) -> bool:
    flags = get_feature_flags(user_id)
    return flags.get(flag_name, DEFAULT_FLAGS.get(flag_name, False))


# ─── Recommendation History ──────────────────────────────────────────────────

def add_recommendation(user_id: str, title: str, description: str,
                       category: str, impact_yearly: float,
                       difficulty: str = "medium") -> str:
    rec_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO recommendation_history (rec_id, user_id, title, description, category, impact_yearly, difficulty, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'suggested', ?)",
        (rec_id, user_id, title, description, category, impact_yearly, difficulty, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return rec_id


def get_recommendations(user_id: str, status: Optional[str] = None) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    query = "SELECT * FROM recommendation_history WHERE user_id = ?"
    params: list = [user_id]
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_recommendation_status(user_id: str, rec_id: str, status: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    resolved = datetime.now().isoformat() if status in ("accepted", "refused") else None
    if resolved:
        c.execute("UPDATE recommendation_history SET status = ?, resolved_at = ? WHERE rec_id = ? AND user_id = ?",
                  (status, resolved, rec_id, user_id))
    else:
        c.execute("UPDATE recommendation_history SET status = ? WHERE rec_id = ? AND user_id = ?",
                  (status, rec_id, user_id))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


# ─── Categories ───────────────────────────────────────────────────────────────

def get_categories() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM categories")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ─── Initialize on import ────────────────────────────────────────────────────

init_db()
seed_categories()

def get_schemes() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM investment_schemes")
    schemes = [dict(r) for r in c.fetchall()]
    conn.close()
    return schemes

def add_scheme(scheme_id: str, name: str, category: str, returns: float, lock_in: int, min_dep: float, risk: str, url: str, fine_print: str, tags: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO investment_schemes 
        (scheme_id, name, category, returns_percentage, lock_in_months, minimum_deposit, risk_level, source_url, fine_print, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (scheme_id, name, category, returns, lock_in, min_dep, risk, url, fine_print, tags))
    conn.commit()
    conn.close()
