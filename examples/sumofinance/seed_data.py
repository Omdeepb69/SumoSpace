"""SumoFinance — Seed Data Generator. 150 realistic transactions over 90 days."""
import random
import db
from datetime import datetime, timedelta

MERCHANTS = {
    "Food": [("Swiggy", 150, 600), ("Zomato", 200, 700), ("Starbucks", 250, 500),
             ("Dominos", 300, 800), ("Chai Point", 50, 150), ("McDonald's", 150, 450)],
    "Transport": [("Uber", 100, 500), ("Ola", 80, 400), ("Rapido", 50, 200),
                  ("Metro Recharge", 200, 500)],
    "Shopping": [("Amazon", 300, 5000), ("Myntra", 500, 3000), ("Flipkart", 400, 4000),
                 ("Croma", 1000, 15000)],
    "Entertainment": [("BookMyShow", 200, 600), ("Spotify", 119, 119),
                      ("Netflix", 499, 499), ("Steam", 200, 2000)],
    "Groceries": [("DMart", 500, 3000), ("BigBasket", 400, 2500), ("Zepto", 200, 1500)],
    "Health": [("Apollo Pharmacy", 100, 800), ("Cult.fit", 1200, 1200)],
    "Bills": [("Airtel", 299, 599), ("Electricity", 800, 2000), ("Water", 200, 400)],
    "Software": [("GitHub", 700, 700), ("Notion", 400, 400)],
}

def seed_user_data(user_id: str):
    """Generate 150 transactions over 90 days with weekend spikes and anomalies."""
    db.clear_user_data(user_id)
    db.update_balance(user_id, 150000)
    
    now = datetime.now()
    random.seed(42)

    # --- Monthly salary & Gig Income ---
    for i in range(3):
        dt = now - timedelta(days=30 * i)
        salary_date = dt.replace(day=1)
        db.add_transaction(user_id, 40000, "Income", "Salary", "income",
                           salary_date.isoformat())
                           
    for i in range(8):
        dt = now - timedelta(days=random.randint(1, 90))
        db.add_transaction(user_id, random.choice([2500, 5000, 1200]), "Income", "Freelance Gig", "income", dt.isoformat())

    # --- Fixed subscriptions ---
    subs = [("Netflix", 499, "Entertainment"), ("Spotify", 119, "Entertainment"),
            ("Cult.fit", 1200, "Health"), ("Airtel", 399, "Bills")]
    for month_offset in range(3):
        for merchant, amount, cat in subs:
            dt = now - timedelta(days=30 * month_offset + random.randint(1, 5))
            db.add_transaction(user_id, amount, cat, merchant, "expense", dt.isoformat())

    # --- Variable spending (main body, ~130 transactions) ---
    txn_count = 0
    for day_offset in range(90):
        dt = now - timedelta(days=day_offset)
        is_weekend = dt.weekday() >= 5
        # More transactions on weekends
        daily_txns = random.randint(2, 4) if is_weekend else random.randint(1, 3)

        for _ in range(daily_txns):
            if txn_count >= 138:
                break
            cat = random.choices(
                ["Food", "Transport", "Shopping", "Entertainment", "Groceries", "Health", "Bills", "Software"],
                weights=[30, 15, 12, 10, 15, 5, 8, 5]
            )[0]
            merchants = MERCHANTS.get(cat, [("Unknown", 100, 500)])
            m_name, m_min, m_max = random.choice(merchants)
            amount = round(random.uniform(m_min, m_max), 0)

            # Weekend spike for Food & Entertainment
            if is_weekend and cat in ("Food", "Entertainment"):
                amount *= 1.2
                amount = round(amount, 0)

            db.add_transaction(user_id, amount, cat, m_name, "expense", dt.isoformat())
            txn_count += 1

    # --- Anomalies (2-3 large purchases) ---
    anomaly_dates = [now - timedelta(days=random.randint(5, 80)) for _ in range(3)]
    anomaly_items = [("Croma", 12500, "Shopping"), ("Amazon", 8900, "Shopping"),
                     ("Zomato", 3200, "Food")]
    for dt, (m, a, c) in zip(anomaly_dates, anomaly_items):
        db.add_transaction(user_id, a, c, m, "expense", dt.isoformat())

    # --- Fixed bills ---
    db.add_fixed_bill(user_id, "Rent", 15000, "monthly")
    db.add_fixed_bill(user_id, "Netflix", 499, "monthly")
    db.add_fixed_bill(user_id, "Spotify", 119, "monthly")
    db.add_fixed_bill(user_id, "Cult.fit", 1200, "monthly")
    db.add_fixed_bill(user_id, "Airtel", 399, "monthly")

    # --- Budgets ---
    db.set_budget(user_id, "Food", 8000)
    db.set_budget(user_id, "Transport", 4000)
    db.set_budget(user_id, "Shopping", 5000)
    db.set_budget(user_id, "Entertainment", 3000)
    db.set_budget(user_id, "Groceries", 6000)

    # --- Goals ---
    db.create_goal(user_id, "Vacation", 50000,
                   (now - timedelta(days=30)).strftime("%Y-%m-%d"),
                   (now + timedelta(days=180)).strftime("%Y-%m-%d"))
    db.create_goal(user_id, "Emergency Fund", 100000,
                   (now - timedelta(days=60)).strftime("%Y-%m-%d"),
                   (now + timedelta(days=365)).strftime("%Y-%m-%d"))

    # --- Feature flags ---
    db.init_feature_flags(user_id)

    print(f"✅ Seeded {txn_count + 3 + 12 + 3} transactions, 5 bills, 5 budgets, 2 goals for user {user_id}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python seed_data.py <user_id>")
        sys.exit(1)
    seed_user_data(sys.argv[1])
