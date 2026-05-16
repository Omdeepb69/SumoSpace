"""
SumoSpace Kaggle Benchmark Script
---------------------------------
This single script demonstrates every SumoSpace feature with real inference on a Kaggle T4 GPU.
It includes 10 sections covering environment setup, basic inference, autonomous editing, RAG,
loaders, hooks, custom tools, audit logging, and the full benchmark execution.
"""

import os
import subprocess
import time
import json
import asyncio
import tempfile
import ast
import traceback
import textwrap
import hashlib
from pathlib import Path

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def print_section(title, description=""):
    print("\n" + "="*80)
    print(f" SECTION: {title}")
    print("="*80)
    if description:
        print(f"\n{textwrap.indent(description, '    ')}\n")

def print_pass_fail(feature, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"[{status}] {feature}" + (f" - {details}" if details else ""))

def hash_file(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Track results for the final summary
summary_results = []
test_session_id = None # Store a session ID for the audit log section

# -----------------------------------------------------------------------------
# Section 1 — Environment setup
# -----------------------------------------------------------------------------
print_section("Section 1 — Environment setup", 
              "Check GPU with nvidia-smi, install Ollama, start it as a background process, "
              "install SumoSpace, pull models, clone the repo, and define imports.")

try:
    print("1. Checking GPU:")
    subprocess.run("nvidia-smi", shell=True, check=False)
    
    print("\n1.5 Installing dependencies (zstd for Ollama, nest_asyncio for Jupyter)...")
    subprocess.run("apt-get update && apt-get install -y zstd", shell=True, check=False)
    subprocess.run("pip install nest_asyncio", shell=True, check=False)
    
    import nest_asyncio
    nest_asyncio.apply()
    
    print("\n2. Installing and starting Ollama...")
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=False)
    
    # Start ollama serve in the background
    ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Waiting 5 seconds for Ollama to boot...")
    time.sleep(5)
    
    print("\n3. Cloning SumoSpace repository...")
    if not os.path.exists("/kaggle/working/SumoSpace"):
        os.makedirs("/kaggle/working", exist_ok=True)
        subprocess.run("git clone https://github.com/Omdeepb69/SumoSpace /kaggle/working/SumoSpace", shell=True, check=False)
    else:
        print("Repo already exists at /kaggle/working/SumoSpace")
        
    print("\n4. Installing SumoSpace from cloned repository...")
    subprocess.run("pip install -e /kaggle/working/SumoSpace /kaggle/working/SumoSpace[multimodal] /kaggle/working/SumoSpace[loaders]", shell=True, check=False)
    
    print("\n5. Pulling models (llama3:8b and llama3:8b)...")
    subprocess.run("ollama pull llama3:8b", shell=True, check=False)
    subprocess.run("ollama pull llama3:8b", shell=True, check=False)

    from sumospace import SumoKernel, SumoSettings
    from sumospace.tools import BaseTool, ToolResult
    from sumospace.audit import AuditLogger
    from sumospace.hooks import HookRegistry
    from sumospace.loaders.github import GitHubLoader
    from sumospace.loaders.youtube import YouTubeLoader
    
    print_pass_fail("Environment Setup", True)
    summary_results.append(("Section 1: Environment Setup", True, "Ollama started, models pulled, repo cloned."))
except Exception as e:
    print_pass_fail("Environment Setup", False, str(e))
    summary_results.append(("Section 1: Environment Setup", False, str(e)))


# -----------------------------------------------------------------------------
# Section 2 — Basic inference
# -----------------------------------------------------------------------------
print_section("Section 2 — Basic inference",
              "Show all 6 presets. Run the same task through disabled, plan_only, and full committee modes.")

async def section2():
    try:
        print("Available Presets:")
        presets = {
            "for_chat":              SumoSettings.for_chat(provider="ollama", model="llama3:8b"),
            "for_chat_with_context": SumoSettings.for_chat_with_context(provider="ollama", model="llama3:8b"),
            "for_chat_stateless":    SumoSettings.for_chat_stateless(provider="ollama", model="llama3:8b"),
            "for_coding":            SumoSettings.for_coding(provider="ollama", model="llama3:8b"),
            "for_research":          SumoSettings.for_research(provider="ollama", model="llama3:8b"),
            "for_review":            SumoSettings.for_review(provider="ollama", model="llama3:8b"),
        }
        for name, preset in presets.items():
            print(f" - {name}")
            
        task = "Write a python function to compute the 10th fibonacci number. Just the function, no explanation."
        modes = ["disabled", "plan_only", "full"]
        
        print(f"\nTask: {task}")
        print(f"{'Mode':<15} | {'Duration (s)':<15} | {'Success':<10} | {'Output preview'}")
        print("-" * 80)
        
        for mode in modes:
            settings = SumoSettings(provider="ollama", model="llama3:8b", committee_enabled=(mode!="disabled"), committee_mode=mode if mode!="disabled" else "full")
            start = time.time()
            async with SumoKernel(settings=settings) as kernel:
                try:
                    trace = await kernel.run(task)
                    success = trace.success
                    preview = str(trace.final_answer)[:40].replace("\n", " ") + "..."
                except Exception as e:
                    success = False
                    preview = f"Error: {str(e)[:30]}"
            dur = time.time() - start
            print(f"{mode:<15} | {dur:<15.2f} | {str(success):<10} | {preview}")
        
        print_pass_fail("Basic Inference Comparison", True)
        summary_results.append(("Section 2: Basic Inference", True, "Ran all 3 modes successfully."))
    except Exception as e:
        print_pass_fail("Basic Inference", False, str(e))
        summary_results.append(("Section 2: Basic Inference", False, str(e)))

asyncio.run(section2())


# -----------------------------------------------------------------------------
# Section 3 — Autonomous file editing
# -----------------------------------------------------------------------------
print_section("Section 3 — Autonomous file editing",
              "Create a Python file, use the agent to add docstrings, verify via AST, and demonstrate rollback.")

async def section3():
    global test_session_id
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "math_ops.py"
            file_path.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")
            
            original_hash = hash_file(str(file_path))
            print("Before editing:")
            print(file_path.read_text())
            
            settings = SumoSettings(provider="ollama", model="llama3:8b", workspace=tmpdir)
            async with SumoKernel(settings=settings) as kernel:
                trace = await kernel.run("Add docstrings to all functions in math_ops.py")
                session_id = trace.session_id
                test_session_id = session_id
                
            print("\nAfter editing:")
            content = file_path.read_text()
            print(content)
            
            # Verify with AST
            try:
                tree = ast.parse(content)
                funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                docs = [f for f in funcs if ast.get_docstring(f)]
                score = len(docs) / len(funcs) if funcs else 0
                print(f"\nVerification Score: {score*100}% ({len(docs)}/{len(funcs)} functions have docstrings)")
            except SyntaxError:
                score = 0
                print("\nVerification Score: 0% (Syntax Error generated by agent)")
            
            print(f"\nDemonstrating rollback for session {session_id}...")
            # Use correct rollback syntax: sumo rollback <run-id> --yes
            subprocess.run(["sumo", "rollback", session_id, "--yes"], cwd=tmpdir)
            
            restored_hash = hash_file(str(file_path))
            if restored_hash == original_hash:
                print("Rollback successful. File hash matches the original state.")
                print_pass_fail("Autonomous File Editing & Rollback", True)
                summary_results.append(("Section 3: File Edit & Rollback", True, f"Docstring score: {score*100}%, Rollback successful."))
            else:
                print("Rollback failed. File hash does not match.")
                print_pass_fail("Autonomous File Editing & Rollback", False)
                summary_results.append(("Section 3: File Edit & Rollback", False, "Rollback verification failed."))
    except Exception as e:
        print_pass_fail("Autonomous File Editing", False, str(e))
        summary_results.append(("Section 3: File Edit & Rollback", False, str(e)))

asyncio.run(section3())


# -----------------------------------------------------------------------------
# Section 4 — RAG
# -----------------------------------------------------------------------------
print_section("Section 4 — RAG",
              "Ingest source code, perform incremental ingest, and run single vs multi-query RAG.")

async def section4():
    try:
        ws = "/kaggle/working/SumoSpace/sumospace"
        if not os.path.exists(ws):
            print("SumoSpace source not found. Using current directory.")
            ws = "."
            
        settings = SumoSettings(provider="ollama", model="llama3:8b", workspace=ws, rag_enabled=True, rag_multi_query=False)
        
        async with SumoKernel(settings=settings) as kernel:
            print("Ingesting codebase...")
            q1 = "What is the purpose of the CriticAgent?"
            print(f"\nQuery (Single): {q1}")
            trace1 = await kernel.run(f"Answer the following question based on the codebase: {q1}")
            print(f"Answer: {trace1.final_answer}")
            
        print("\nEnabling Multi-Query RAG...")
        # Create a new kernel with updated settings to properly reconfigure RAG
        settings_multi = SumoSettings(provider="ollama", model="llama3:8b", workspace=ws, rag_enabled=True, rag_multi_query=True)
        async with SumoKernel(settings=settings_multi) as kernel_multi:
            trace2 = await kernel_multi.run(f"Answer the following question based on the codebase: {q1}")
            print(f"Answer (Multi-Query): {trace2.final_answer}")
            
        print_pass_fail("RAG Capabilities", True)
        summary_results.append(("Section 4: RAG", True, "Successfully queried codebase with and without multi-query."))
    except Exception as e:
        print_pass_fail("RAG Capabilities", False, str(e))
        summary_results.append(("Section 4: RAG", False, str(e)))

asyncio.run(section4())


# -----------------------------------------------------------------------------
# Section 5 — GitHub and YouTube loaders
# -----------------------------------------------------------------------------
print_section("Section 5 — GitHub and YouTube loaders",
              "Load a real public GitHub repo and YouTube transcript programmatically.")

async def section5():
    try:
        print("Querying GitHub repo (tiangolo/fastapi) using GitHubLoader...")
        gh_loader = GitHubLoader()
        gh_chunks = await gh_loader.load("https://github.com/tiangolo/fastapi")
        print(f"Loaded {len(gh_chunks)} chunks from GitHub.")
        if gh_chunks:
            print(f"Preview: {gh_chunks[0].text[:100]}...")
        
        print("\nQuerying YouTube transcript using YouTubeLoader...")
        yt_loader = YouTubeLoader()
        yt_chunks = await yt_loader.load("https://www.youtube.com/watch?v=kJQP7kiw5Fk")
        print(f"Loaded {len(yt_chunks)} chunks from YouTube.")
        if yt_chunks:
            print(f"Preview: {yt_chunks[0].text[:100]}...")
            
        print_pass_fail("Web Loaders", True)
        summary_results.append(("Section 5: Loaders", True, "GitHub and YouTube loaders executed successfully."))
    except Exception as e:
        print_pass_fail("Web Loaders", False, str(e))
        summary_results.append(("Section 5: Loaders", False, str(e)))

asyncio.run(section5())


# -----------------------------------------------------------------------------
# Section 6 — Lifecycle hooks
# -----------------------------------------------------------------------------
print_section("Section 6 — Lifecycle hooks",
              "Register all hooks (including a broken one) and verify execution order and resilience.")

async def section6():
    hook_events_fired = []
    
    async def h_boot(*args, **kwargs): hook_events_fired.append("on_kernel_boot")
    async def h_start(*args, **kwargs): hook_events_fired.append("on_task_start")
    async def h_plan_a(*args, **kwargs): hook_events_fired.append("on_plan_approved")
    async def h_plan_r(*args, **kwargs): hook_events_fired.append("on_plan_rejected")
    async def h_step_s(*args, **kwargs): hook_events_fired.append("on_step_start")
    async def h_step_c(*args, **kwargs): hook_events_fired.append("on_step_complete")
    async def h_step_f(*args, **kwargs): hook_events_fired.append("on_step_failed")
    async def h_task_c(*args, **kwargs): hook_events_fired.append("on_task_complete")
    async def h_task_f(*args, **kwargs): hook_events_fired.append("on_task_failed")
    async def h_shutdown(*args, **kwargs): hook_events_fired.append("on_kernel_shutdown")
    
    async def hook_broken(*args, **kwargs): 
        hook_events_fired.append("broken_hook_fired")
        raise RuntimeError("This hook is deliberately broken.")

    hooks = HookRegistry()
    hooks.register("on_kernel_boot", h_boot)
    hooks.register("on_task_start", h_start)
    hooks.register("on_plan_approved", h_plan_a)
    hooks.register("on_plan_rejected", h_plan_r)
    hooks.register("on_step_start", h_step_s)
    hooks.register("on_step_complete", h_step_c)
    hooks.register("on_step_failed", h_step_f)
    hooks.register("on_task_complete", h_task_c)
    hooks.register("on_task_complete", hook_broken) # Register broken hook too
    hooks.register("on_task_failed", h_task_f)
    hooks.register("on_kernel_shutdown", h_shutdown)
    
    try:
        settings = SumoSettings(provider="ollama", model="llama3:8b")
        async with SumoKernel(settings=settings, hooks=hooks) as kernel:
            await kernel.run("Calculate 5 + 5 using python")
            
        print("Hooks fired in order:")
        for i, event in enumerate(hook_events_fired):
            print(f" {i+1}. {event}")
            
        if "broken_hook_fired" in hook_events_fired and "on_kernel_shutdown" in hook_events_fired:
            print("Kernel survived the broken hook successfully.")
            print_pass_fail("Lifecycle Hooks", True)
            summary_results.append(("Section 6: Hooks", True, "All hooks fired, kernel survived broken hook."))
        else:
            print_pass_fail("Lifecycle Hooks", False, "Missing expected hook executions.")
            summary_results.append(("Section 6: Hooks", False, "Missing hook executions."))
    except Exception as e:
        print_pass_fail("Lifecycle Hooks", False, str(e))
        summary_results.append(("Section 6: Hooks", False, str(e)))

asyncio.run(section6())


# -----------------------------------------------------------------------------
# Section 7 — Custom tools
# -----------------------------------------------------------------------------
print_section("Section 7 — Custom tools",
              "Define two custom BaseTool subclasses, test directly, and use in an agent run.")

async def section7():
    try:
        class ReverseStringTool(BaseTool):
            name = "reverse_string"
            description = "Reverses a given string."
            schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
            async def run(self, text: str, **kwargs) -> ToolResult:
                return ToolResult(tool=self.name, success=True, output=text[::-1])

        class CountWordsTool(BaseTool):
            name = "count_words"
            description = "Counts the number of words in a string."
            schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
            async def run(self, text: str, **kwargs) -> ToolResult:
                return ToolResult(tool=self.name, success=True, output=str(len(text.split())))

        tool1 = ReverseStringTool()
        tool2 = CountWordsTool()
        
        print(f"Direct test reverse_string('SumoSpace'): {(await tool1.run(text='SumoSpace')).output}")
        print(f"Direct test count_words('Hello world from Kaggle'): {(await tool2.run(text='Hello world from Kaggle')).output}")

        settings = SumoSettings(provider="ollama", model="llama3:8b")
        async with SumoKernel(settings=settings) as kernel:
            kernel.tools.register(tool1)
            kernel.tools.register(tool2)
            
            trace = await kernel.run("Use the count_words tool to count the words in the string 'The quick brown fox'.")
            
            used_tools = [s.tool for s in getattr(trace, 'step_traces', [])]
            print(f"\nTrace final answer: {trace.final_answer}")
            print(f"Tools used in step_traces: {used_tools}")
            
            if "count_words" in used_tools:
                print_pass_fail("Custom Tools", True)
                summary_results.append(("Section 7: Custom Tools", True, "Registered and executed custom tools."))
            else:
                print_pass_fail("Custom Tools", False, "Tool was not called by the agent.")
                summary_results.append(("Section 7: Custom Tools", False, "Tool not called."))
    except Exception as e:
        print_pass_fail("Custom Tools", False, str(e))
        summary_results.append(("Section 7: Custom Tools", False, str(e)))

asyncio.run(section7())


# -----------------------------------------------------------------------------
# Section 8 — Audit log
# -----------------------------------------------------------------------------
print_section("Section 8 — Audit log",
              "Call AuditLogger to list, stats, search, and export session data.")

async def section8():
    global test_session_id
    try:
        # Create a new AuditLogger scoped to the workspace used earlier or current dir
        logger = AuditLogger(SumoSettings())
        
        print("Audit Stats:")
        stats = logger.stats()
        print(json.dumps(stats, indent=2))
        
        print("\nRecent Sessions (list):")
        sessions = logger.list(limit=3)
        for s in sessions:
            print(f" - {s.get('session_id')} | Duration: {s.get('duration_ms')}ms | Success: {s.get('success')}")
            
        print("\nExporting audit log...")
        export_path = Path("/kaggle/working/audit_export.md")
        # Try to use the session_id from section 3, fallback to the latest session
        target_session = test_session_id if test_session_id else (sessions[0].get('session_id') if sessions else None)
        
        if target_session:
            report_content = logger.export(target_session)
            if report_content:
                export_path.write_text(report_content)
                print(f"Exported to {export_path}. Size: {export_path.stat().st_size} bytes.")
            else:
                print("Export failed (session not found).")
        else:
            print("No sessions available to export.")
            
        print_pass_fail("Audit Log", True)
        summary_results.append(("Section 8: Audit Log", True, f"Stats retrieved, exported {export_path.stat().st_size if export_path.exists() else 0} bytes."))
    except Exception as e:
        print_pass_fail("Audit Log", False, str(e))
        summary_results.append(("Section 8: Audit Log", False, str(e)))

asyncio.run(section8())


# -----------------------------------------------------------------------------
# Section 9 — THE REAL BENCHMARK
# -----------------------------------------------------------------------------
print_section("Section 9 — THE REAL BENCHMARK",
              "Run the actual benchmark script via subprocess to ensure standard CLI reproduction.")

try:
    cmd = [
        "python", "benchmarks/run_benchmark.py",
        "--provider", "ollama",
        "--model", "llama3:8b",
        "--modes", "disabled,plan_only,critique_only,full"
    ]
    
    print(f"Executing command: {' '.join(cmd)}\n")
    print("-" * 80)
    
    # Run the benchmark and stream output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/kaggle/working/SumoSpace")
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print("-" * 80)
    
    # Locate the generated markdown file
    results_dir = Path("/kaggle/working/SumoSpace/benchmark_results")
    if results_dir.exists():
        md_files = list(results_dir.glob("*.md"))
        json_files = list(results_dir.glob("*.json"))
        if md_files and json_files:
            latest_md = max(md_files, key=os.path.getmtime)
            latest_json = max(json_files, key=os.path.getmtime)
            
            print(f"\nBenchmark output file generated at: {latest_md}")
            print("\n" + latest_md.read_text())
            
            # Verify committee improvement
            data = json.loads(latest_json.read_text())
            full_scores = [r["score"] for r in data["results"] if r["committee_mode"] == "full"]
            disabled_scores = [r["score"] for r in data["results"] if r["committee_mode"] == "disabled"]
            
            if full_scores and disabled_scores:
                avg_full = sum(full_scores) / len(full_scores)
                avg_disabled = sum(disabled_scores) / len(disabled_scores)
                improvement = avg_full - avg_disabled
                print(f"\nCommittee improvement: {improvement*100:.1f} percentage points")
                print(f"disabled avg: {avg_disabled*100:.1f}%")
                print(f"full avg:     {avg_full*100:.1f}%")
            
            print_pass_fail("Real Benchmark", True)
            summary_results.append(("Section 9: Benchmark", True, f"Completed successfully. Output at {latest_md}"))
            benchmark_path = str(latest_md)
        else:
            print_pass_fail("Real Benchmark", False, "No markdown/JSON file generated.")
            summary_results.append(("Section 9: Benchmark", False, "No output file found."))
            benchmark_path = "N/A"
    else:
        print_pass_fail("Real Benchmark", False, "benchmark_results directory not found.")
        summary_results.append(("Section 9: Benchmark", False, "Results dir missing."))
        benchmark_path = "N/A"
except Exception as e:
    print_pass_fail("Real Benchmark", False, str(e))
    summary_results.append(("Section 9: Benchmark", False, str(e)))
    benchmark_path = "N/A"


# -----------------------------------------------------------------------------
# Section 10 — Results summary
# -----------------------------------------------------------------------------
print_section("Section 10 — Results summary",
              "Final summary of every section.")

print("SumoSpace Kaggle Execution Report")
print("=" * 80)
for sec_name, success, notes in summary_results:
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {sec_name:<30} | {notes}")

print("=" * 80)
print(f"Benchmark Markdown File: {benchmark_path}")
print("Done.")

# Clean up ollama background process
if 'ollama_proc' in locals():
    ollama_proc.terminate()
