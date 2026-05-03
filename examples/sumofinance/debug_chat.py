import asyncio
from finance_kernel import FinanceKernel

async def main():
    fk = FinanceKernel("71fcdd40-357a-49be-8cc0-4f05b48d3870", "test_session")
    print("Booting...")
    await fk.boot()
    print("Booted.")
    
    print("--- Q1 ---")
    resp1 = await fk.chat("What is my balance?")
    print("A1:", resp1[:100])
    
    print("--- Q2 ---")
    resp2 = await fk.chat("What are my goals?")
    print("A2:", resp2[:100])

asyncio.run(main())
