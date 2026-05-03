import asyncio
from finance_kernel import FinanceKernel

async def main():
    fk = FinanceKernel("71fcdd40-357a-49be-8cc0-4f05b48d3870", "test_session_new")
    await fk.boot()
    print("Booted.")
    
    # Bypass the try-except to see the real exception
    msg = "hi"
    try:
        r1 = await fk.chat(msg)
        print("Success:", len(r1))
    except Exception as e:
        print("REAL ERROR:", type(e), str(e))
        import traceback
        traceback.print_exc()

asyncio.run(main())
