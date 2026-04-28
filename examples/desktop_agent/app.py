# examples/desktop_agent/app.py

"""
SumoSpace Desktop Agent — Streamlit UI
========================================
A visual control panel for the autonomous desktop agent.
Shows live screenshots, action history, and agent reasoning.

Run:
    streamlit run examples/desktop_agent/app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure sumospace is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentConfig, DesktopAgent, StepResult, TaskResult


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SumoSpace Desktop Agent",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main > div { padding-top: 1rem; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header */
    .agent-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .agent-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .agent-header p {
        margin: 0.3rem 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Step cards */
    .step-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }
    .step-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        background: rgba(255,255,255,0.06);
    }

    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 8px;
        padding: 0.15rem 0.6rem;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }

    .tool-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border-radius: 6px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.4rem;
    }

    .success-badge {
        color: #4ade80;
    }
    .fail-badge {
        color: #f87171;
    }

    /* Stats row */
    .stat-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Screenshot container */
    .screenshot-container {
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Thinking indicator */
    .thinking {
        background: rgba(255, 193, 7, 0.1);
        border-left: 3px solid #ffc107;
        padding: 0.5rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        margin-bottom: 0.5rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── State Init ───────────────────────────────────────────────────────────────

if "steps" not in st.session_state:
    st.session_state.steps = []
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None
if "task_history" not in st.session_state:
    st.session_state.task_history = []


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="agent-header">
    <h1>🖥️ SumoSpace Desktop Agent</h1>
    <p>Autonomous desktop automation powered by SumoSpace multi-agent orchestration</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")

    provider = st.selectbox(
        "LLM Provider",
        ["ollama", "hf", "gemini", "openai", "anthropic"],
        index=0,
        help="AI brain for planning actions",
    )

    model = st.text_input(
        "Model",
        value="default",
        help="Model name or alias (e.g. phi3:mini, gemini-1.5-flash)",
    )

    max_steps = st.slider("Max Steps", 5, 30, 15, help="Safety limit per task")
    step_delay = st.slider("Step Delay (s)", 0.0, 3.0, 1.0, 0.5,
                           help="Pause between actions")

    st.markdown("---")
    st.markdown("### 📊 SumoSpace Features Used")
    st.markdown("""
    - **ProviderRouter** → AI brain
    - **MemoryManager** → Action history
    - **ScopeManager** → Session isolation
    - **ToolRegistry** → 14 desktop tools
    """)

    st.markdown("---")
    st.markdown("### 🔧 Desktop Tools")
    tools_list = [
        "📸 take_screenshot", "📖 read_screen", "🖱️ click_at",
        "🖱️ double_click", "🖱️ right_click", "⌨️ type_text",
        "⌨️ hotkey", "⌨️ press_key", "↗️ move_mouse",
        "📜 scroll", "↔️ drag_to", "🚀 open_app",
        "⏱️ wait", "🔍 locate_on_screen",
    ]
    for t in tools_list:
        st.markdown(f"<small>{t}</small>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.task_history:
        st.markdown("### 📜 Task History")
        for i, th in enumerate(reversed(st.session_state.task_history[-5:])):
            status = "✅" if th.get("success") else "❌"
            st.markdown(
                f"<small>{status} {th['task'][:40]}... "
                f"({th['steps']} steps)</small>",
                unsafe_allow_html=True,
            )


# ─── Main Content ────────────────────────────────────────────────────────────

# Task input
col_input, col_btn = st.columns([4, 1])
with col_input:
    task = st.text_input(
        "What should the agent do?",
        placeholder="Open VS Code and open the sumospace project from the projects folder...",
        label_visibility="collapsed",
    )
with col_btn:
    run_button = st.button("🚀 Execute", type="primary", use_container_width=True,
                           disabled=st.session_state.running)

# Example tasks
with st.expander("💡 Example Tasks", expanded=not bool(st.session_state.steps)):
    ex_cols = st.columns(3)
    examples = [
        "Open VS Code and create a new Python file",
        "Open the file manager and navigate to Documents",
        "Open Firefox and go to github.com",
        "Open the terminal and run 'ls -la'",
        "Take a screenshot and tell me what's on screen",
        "Open the text editor and write 'Hello from SumoSpace!'",
    ]
    for i, ex in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                task = ex
                run_button = True


# ─── Execution ────────────────────────────────────────────────────────────────

def run_agent(task_text: str, provider: str, model: str,
              max_steps: int, step_delay: float, on_step=None) -> TaskResult:
    """Run the desktop agent synchronously (for Streamlit)."""

    config = AgentConfig(
        provider=provider,
        model=model,
        max_steps=max_steps,
        verbose=False,
        step_delay=step_delay,
    )

    agent = DesktopAgent(config)

    async def _run():
        await agent.initialize()
        try:
            return await agent.execute(task_text, on_step=on_step)
        finally:
            await agent.cleanup()

    return asyncio.run(_run())


def render_step(step: StepResult):
    """Render a single step card."""
    status_icon = "✅" if step.tool_result.success else "❌"
    tool_name = step.action.tool

    with st.container():
        st.markdown(f"""
        <div class="step-card">
            <span class="step-number">Step {step.step}</span>
            <span class="tool-badge">{tool_name}</span>
            <span class="{'success-badge' if step.tool_result.success else 'fail-badge'}">
                {status_icon}
            </span>
            <span style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">
                {step.duration_ms:.0f}ms
            </span>
            <div class="thinking">{step.action.reasoning}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show screenshot if this step captured one
        if step.screenshot_path and Path(step.screenshot_path).exists():
            with st.expander(f"📸 Screenshot from Step {step.step}"):
                st.image(step.screenshot_path, use_container_width=True)

        # Show OCR text if available
        if step.screen_text:
            with st.expander(f"📖 Screen Text from Step {step.step}"):
                st.code(step.screen_text[:2000], language=None)

        # Show tool output
        if step.tool_result.output and step.action.tool != "take_screenshot":
            with st.expander(f"📄 Output: {step.tool_result.output[:60]}"):
                st.text(step.tool_result.output)

        if step.tool_result.error:
            st.error(f"Error: {step.tool_result.error}")


if (run_button or task) and task and not st.session_state.running:
    # Only run if the button was actually pressed
    if not run_button:
        st.stop()

    st.session_state.running = True
    st.session_state.steps = []
    st.session_state.result = None

    with st.status("🤖 Agent executing...", expanded=True) as status:
        st.write(f"**Task:** {task}")
        st.write(f"**Provider:** {provider} / {model}")
        
        st.markdown("### 📋 Live Action Timeline")
        steps_container = st.container()
        
        def on_step(step: StepResult):
            with steps_container:
                render_step(step)

        try:
            result = run_agent(task, provider, model, max_steps, step_delay, on_step=on_step)

            st.session_state.result = result
            st.session_state.steps = result.steps
            st.session_state.task_history.append({
                "task": task,
                "success": result.success,
                "steps": len(result.steps),
                "time_ms": result.total_duration_ms,
            })

            if result.success:
                status.update(label="✅ Task completed!", state="complete")
            else:
                status.update(label="⚠️ Task incomplete", state="error")

        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")
            st.error(f"Agent failed: {e}")

        finally:
            st.session_state.running = False


# ─── Results Display ──────────────────────────────────────────────────────────

if st.session_state.result:
    result = st.session_state.result

    # Stats row
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{len(result.steps)}</div>
            <div class="stat-label">Steps Taken</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        success_count = sum(1 for s in result.steps if s.tool_result.success)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{success_count}</div>
            <div class="stat-label">Successful</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{result.total_duration_ms/1000:.1f}s</div>
            <div class="stat-label">Total Time</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{'✓' if result.success else '✗'}</div>
            <div class="stat-label">Status</div>
        </div>""", unsafe_allow_html=True)

    # Summary
    if result.summary:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Summary:** {result.summary}")

    # Session info
    st.markdown("---")
    st.markdown(
        f"<small style='color: rgba(255,255,255,0.3);'>"
        f"Session: {result.session_id} | "
        f"Provider: {provider}/{model} | "
        f"Scope: session-level isolation via ScopeManager"
        f"</small>",
        unsafe_allow_html=True,
    )

elif not st.session_state.running:
    # Empty state
    st.markdown("<br>" * 2, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; opacity: 0.5;">
            <div style="font-size: 4rem;">🤖</div>
            <p style="font-size: 1.1rem; margin-top: 1rem;">
                Enter a task above and click <b>Execute</b> to start
            </p>
            <p style="font-size: 0.8rem;">
                The agent will autonomously control your desktop
            </p>
        </div>
        """, unsafe_allow_html=True)
