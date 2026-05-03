import asyncio
from finance_kernel import FinanceKernel

async def main():
    fk = FinanceKernel("71fcdd40-357a-49be-8cc0-4f05b48d3870", "test_session_new")
    await fk.boot()
    print("Booted. Going to run chat...")
    
    msg = "What is my balance?"
    print("P1: detect_intent")
    
    print("P1.1: Exact Cache")
    
    print("P1.5: Semantic Cache")
    if fk.cache_col:
        print("Running cache_col.query...")
        cache_res = fk.cache_col.query(query_texts=[msg], n_results=1)
        print("Query done.")
        
    print("P2: Build context")
    live_context = await asyncio.to_thread(fk._build_live_context)
    print("Context done.")
    
    print("P3: Ollama")
    from finance_kernel import _ollama_complete
    res = await _ollama_complete(user=msg)
    print("Ollama done:", len(res))

asyncio.run(main())
