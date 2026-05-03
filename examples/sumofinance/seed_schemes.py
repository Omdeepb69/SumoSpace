
import asyncio
import os
import sys
import uuid
from finance_kernel import FinanceKernel
import db

# Sample schemes with descriptions for VDB ingestion
SCHEMES_KNOWLEDGE = [
    {
        "name": "Public Provident Fund (PPF)",
        "type": "Government",
        "return_rate": "7.1%",
        "lock_in": "15 years",
        "min_amount": 500,
        "description": "PPF is a long-term, government-backed savings scheme in India. It offers tax-free returns under Section 80C. Ideal for retirement planning and safe long-term wealth creation. Premature withdrawal is allowed after 7 years under specific conditions."
    },
    {
        "name": "Nippon India Liquid Fund",
        "type": "Debt Mutual Fund",
        "return_rate": "6.8% - 7.2%",
        "lock_in": "None",
        "min_amount": 500,
        "description": "Liquid funds are excellent for parking surplus cash for short periods (1-6 months). They offer higher returns than savings accounts with high liquidity. They invest in short-term market instruments like Treasury Bills."
    },
    {
        "name": "Sukanya Samriddhi Yojana (SSY)",
        "type": "Government",
        "return_rate": "8.2%",
        "lock_in": "21 years",
        "min_amount": 250,
        "description": "A government scheme specifically for the girl child. It offers one of the highest interest rates among small savings schemes. Maturity occurs after 21 years or upon marriage after age 18. Tax-exempt at all levels."
    },
    {
        "name": "HDFC Mid-Cap Opportunities Fund",
        "type": "Equity Mutual Fund",
        "return_rate": "12% - 18%",
        "lock_in": "None",
        "min_amount": 100,
        "description": "Focuses on stocks of mid-sized companies with high growth potential. Higher risk than large-cap funds but offers significantly higher long-term returns. Recommended for a 5-7 year horizon."
    },
    {
        "name": "SBI Bluechip Fund",
        "type": "Equity Mutual Fund",
        "return_rate": "10% - 13%",
        "lock_in": "None",
        "min_amount": 5000,
        "description": "Invests in large-cap companies with a proven track record. Offers stable growth with relatively lower volatility compared to mid or small-cap funds. Good for core portfolio building."
    },
    {
        "name": "Post Office Time Deposit",
        "type": "Government",
        "return_rate": "7.5%",
        "lock_in": "5 years",
        "min_amount": 1000,
        "description": "Similar to bank fixed deposits but backed by the Indian Post Office. The 5-year deposit qualifies for tax deduction under Section 80C."
    },
    {
        "name": "National Savings Certificate (NSC)",
        "type": "Government",
        "return_rate": "7.7%",
        "lock_in": "5 years",
        "min_amount": 1000,
        "description": "A fixed-income post office savings scheme. Interest is compounded annually but paid only at maturity. Extremely safe and popular for tax saving."
    },
    {
        "name": "Gold BeES (ETF)",
        "type": "Gold",
        "return_rate": "9% - 11%",
        "lock_in": "None",
        "min_amount": 50,
        "description": "An Exchange Traded Fund that tracks the domestic price of real gold. It allows you to invest in gold in small quantities without worrying about purity or storage. Highly liquid."
    },
    {
        "name": "Senior Citizens Savings Scheme (SCSS)",
        "type": "Government",
        "return_rate": "8.2%",
        "lock_in": "5 years",
        "min_amount": 1000,
        "description": "Designed for individuals above 60 years. Offers quarterly interest payments, providing a steady income stream. Safe and offers tax benefits."
    },
    {
        "name": "Quant Small Cap Fund",
        "type": "Equity Mutual Fund",
        "return_rate": "15% - 25%",
        "lock_in": "None",
        "min_amount": 1000,
        "description": "Invests in small-sized companies. High volatility but potential for explosive growth. Suitable for aggressive investors with a 7-10 year time horizon."
    },
    {
        "name": "Axis ELSS Tax Saver Fund",
        "type": "Equity Mutual Fund",
        "return_rate": "12% - 15%",
        "lock_in": "3 years",
        "min_amount": 500,
        "description": "Equity Linked Savings Scheme that offers tax deduction under Section 80C. It has the shortest lock-in period (3 years) among all tax-saving options."
    }
]

async def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not user_id:
        print("Usage: python seed_schemes.py <user_id>")
        return

    fk = FinanceKernel(user_id=user_id)
    await fk.boot()

    print(f"Ingesting {len(SCHEMES_KNOWLEDGE)} schemes into VDB for user {user_id}...")

    for s in SCHEMES_KNOWLEDGE:
        text = f"Scheme: {s['name']}\nType: {s['type']}\nReturns: {s['return_rate']}\nLock-in: {s['lock_in']}\nMin Amount: ₹{s['min_amount']}\nDetails: {s['description']}"
        metadata = {"type": "financial_scheme", "scheme_name": s["name"], "risk": "low" if "Government" in s["type"] else "high"}
        
        # Add to SQLite for deterministic tool discovery
        db.add_scheme(
            scheme_id=str(uuid.uuid4()),
            name=s["name"], 
            category=s["type"], 
            returns=float(s["return_rate"].replace("%","").split("-")[-1].strip()), 
            lock_in=0 if "None" in s["lock_in"] else int(s["lock_in"].split()[0]), 
            min_dep=float(s["min_amount"]),
            risk="Low" if "Government" in s["type"] else "High",
            url="https://finance.gov.in/schemes",
            fine_print="Terms and conditions apply as per government/AMC guidelines.",
            tags=f"{s['type'].lower()}, investment, savings"
        )
        
        # Add to VDB for Chatbot explanation
        await fk._ingest_text(user_id, text, metadata)
        print(f"  - Ingested: {s['name']}")

    print("✅ All schemes ingested and synced to SQLite!")

if __name__ == "__main__":
    asyncio.run(main())
