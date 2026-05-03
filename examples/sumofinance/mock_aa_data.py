import json
import random
from datetime import datetime, timedelta

def generate_mock_aa_transactions(user_id: str, days: int = 30) -> list[dict]:
    """
    Generates mock ReBIT Account Aggregator (AA) transaction data.
    """
    transactions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    merchants = [
        {"name": "Amazon", "type": "DEBIT", "amount_range": (500, 5000), "category": "Shopping"},
        {"name": "Swiggy", "type": "DEBIT", "amount_range": (200, 1500), "category": "Food"},
        {"name": "Netflix", "type": "DEBIT", "amount_range": (649, 649), "category": "Entertainment"},
        {"name": "Uber", "type": "DEBIT", "amount_range": (150, 800), "category": "Transport"},
        {"name": "Salary", "type": "CREDIT", "amount_range": (80000, 80000), "category": "Income"},
        {"name": "Gym", "type": "DEBIT", "amount_range": (2000, 2000), "category": "Health"},
        {"name": "Claude Subscription", "type": "DEBIT", "amount_range": (1800, 1800), "category": "Software"},
        {"name": "Dmart", "type": "DEBIT", "amount_range": (1500, 6000), "category": "Groceries"}
    ]
    
    # Generate some regular salary
    transactions.append({
        "txnId": f"TXN-{random.randint(100000, 999999)}",
        "date": start_date.strftime("%Y-%m-%d"),
        "amount": 80000,
        "type": "CREDIT",
        "narration": "Salary Credit",
        "merchant": "Salary"
    })

    current_date = start_date + timedelta(days=1)
    while current_date <= end_date:
        # 30% chance of a transaction on a given day
        if random.random() < 0.3:
            merchant = random.choice(merchants[:-1]) # Exclude Salary
            amount = round(random.uniform(*merchant["amount_range"]), 2)
            transactions.append({
                "txnId": f"TXN-{random.randint(100000, 999999)}",
                "date": current_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "type": merchant["type"],
                "narration": f"UPI/{merchant['name']}/Purchase",
                "merchant": merchant["name"]
            })
            
        # Specific sub on specific days
        if current_date.day == 5:
             transactions.append({
                "txnId": f"TXN-{random.randint(100000, 999999)}",
                "date": current_date.strftime("%Y-%m-%d"),
                "amount": 1800,
                "type": "DEBIT",
                "narration": "UPI/Claude Subscription",
                "merchant": "Claude Subscription"
            })
             
        current_date += timedelta(days=1)
        
    return transactions

def get_mock_aa_json(user_id: str) -> str:
    return json.dumps(generate_mock_aa_transactions(user_id), indent=2)

if __name__ == "__main__":
    print(get_mock_aa_json("test_user_1"))
