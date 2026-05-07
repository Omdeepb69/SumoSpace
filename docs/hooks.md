# Hooks

Hooks let you inject custom logic at every stage of an agent run — without modifying the framework. Every hook is an **async callable** registered via `SumoSettings`.

---

## Registering a Hook

```python
from sumospace import SumoKernel, SumoSettings

async def my_hook(event_data: dict) -> None:
    print(event_data)

async with SumoKernel(SumoSettings(
    provider="ollama",
    model="phi3:mini",
    hooks={
        "on_plan_created": my_hook,
        "on_task_complete": my_hook,
    }
)) as kernel:
    await kernel.run("...")
```

---

## Hook Reference

### `on_task_start`

**When it fires:** Immediately after `kernel.run(task)` is called, before classification.

**Signature:**
```python
async def on_task_start(task: str, session_id: str) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `task` | `str` | The raw task string passed by the user |
| `session_id` | `str` | UUID for this agent run |

**Use case:** Log task start time, send a notification.

---

### `on_plan_created`

**When it fires:** After the Planner produces a plan, before the Critic reviews it.

**Signature:**
```python
async def on_plan_created(plan: dict, session_id: str) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `plan` | `dict` | The raw plan dict: `{"reasoning": ..., "steps": [...]}` |
| `session_id` | `str` | UUID for this agent run |

**Use case:** Human approval gate, plan logging.

---

### `on_plan_approved`

**When it fires:** After the Critic approves the plan, before any tool is executed.

**Signature:**
```python
async def on_plan_approved(plan: dict, session_id: str) -> None: ...
```

**Use case:** Final confirmation before destructive operations.

---

### `on_plan_rejected`

**When it fires:** When the Critic rejects a plan. May fire multiple times if the Resolver retries.

**Signature:**
```python
async def on_plan_rejected(plan: dict, reason: str, attempt: int, session_id: str) -> None: ...
```

| Parameter | Type | Description |
|---|---|---|
| `plan` | `dict` | The rejected plan |
| `reason` | `str` | The Critic's rejection reason |
| `attempt` | `int` | Which retry this is (1-indexed) |

---

### `on_tool_call`

**When it fires:** Before each individual tool is executed.

**Signature:**
```python
async def on_tool_call(tool: str, args: dict, step: int, session_id: str) -> None: ...
```

| Parameter | Type | Description |
|---|---|---|
| `tool` | `str` | Tool name (e.g. `"write_file"`) |
| `args` | `dict` | Tool arguments |
| `step` | `int` | Step index (0-indexed) |

---

### `on_tool_result`

**When it fires:** After each tool execution, whether successful or not.

**Signature:**
```python
async def on_tool_result(
    tool: str, args: dict, result: str, error: str | None,
    duration_ms: int, step: int, session_id: str
) -> None: ...
```

---

### `on_task_complete`

**When it fires:** After all steps execute successfully and the trace is built.

**Signature:**
```python
async def on_task_complete(trace: dict, session_id: str) -> None: ...
```

| Parameter | Description |
|---|---|
| `trace` | Serialized `AgentTrace` dict |

---

### `on_task_failed`

**When it fires:** If any unhandled error occurs during execution, or if the committee cycle exhausts retries.

**Signature:**
```python
async def on_task_failed(error: str, session_id: str) -> None: ...
```

---

## Complete Examples

### Example 1 — Human Approval Gate

Pause execution after the plan is created and require explicit `y` confirmation.

```python title="hooks/approval_gate.py"
import asyncio
from sumospace import SumoKernel, SumoSettings


async def human_approval_gate(plan: dict, session_id: str) -> None:
    """Prints the plan and waits for y/N before tools execute."""
    print("\n" + "=" * 60)
    print(f"[SumoSpace] Plan for session {session_id}")
    print("=" * 60)
    print(f"\nReasoning:\n  {plan.get('reasoning', '')}\n")
    print("Steps:")
    for i, step in enumerate(plan.get("steps", []), 1):
        args_str = ", ".join(f"{k}={v!r}" for k, v in step.get("args", {}).items())
        print(f"  {i}. {step['tool']}({args_str})")

    print("\n" + "-" * 60)

    # Run input in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None,
        lambda: input("\nApprove this plan? [y/N]: ").strip().lower()
    )

    if answer != "y":
        raise RuntimeError("[SumoSpace] Plan rejected by human operator.")


async def main():
    async with SumoKernel(SumoSettings(
        provider="ollama",
        model="phi3:mini",
        hooks={"on_plan_approved": human_approval_gate},
    )) as kernel:
        trace = await kernel.run("Add type hints to ./src/utils.py")
        print(trace.final_answer)


asyncio.run(main())
```

---

### Example 2 — Slack Notification

Post a message to Slack when a task completes (or fails).

```python title="hooks/slack_notify.py"
import httpx
import asyncio
import json
from sumospace import SumoKernel, SumoSettings

SLACK_WEBHOOK = "https://hooks.slack.com/services/T.../B.../..."


async def notify_slack_complete(trace: dict, session_id: str) -> None:
    duration_s = trace.get("duration_ms", 0) / 1000
    text = (
        f":white_check_mark: *SumoSpace task complete*\n"
        f"Session: `{session_id}`\n"
        f"Duration: `{duration_s:.1f}s`\n"
        f"Answer: {trace.get('final_answer', 'N/A')}"
    )
    async with httpx.AsyncClient() as client:
        await client.post(SLACK_WEBHOOK, json={"text": text})


async def notify_slack_failed(error: str, session_id: str) -> None:
    text = (
        f":x: *SumoSpace task failed*\n"
        f"Session: `{session_id}`\n"
        f"Error: ```{error}```"
    )
    async with httpx.AsyncClient() as client:
        await client.post(SLACK_WEBHOOK, json={"text": text})


async def main():
    async with SumoKernel(SumoSettings(
        provider="ollama",
        model="phi3:mini",
        hooks={
            "on_task_complete": notify_slack_complete,
            "on_task_failed": notify_slack_failed,
        },
    )) as kernel:
        trace = await kernel.run("Refactor auth.py to use async/await")
        print(trace.final_answer)


asyncio.run(main())
```

---

### Example 3 — Cost & Duration Tracking

Log every task to a local SQLite database for trend analysis.

```python title="hooks/cost_tracker.py"
import aiosqlite
import asyncio
from pathlib import Path
from sumospace import SumoKernel, SumoSettings

DB_PATH = Path("~/.sumo_telemetry.db").expanduser()


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS task_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                intent      TEXT,
                model       TEXT,
                duration_ms INTEGER,
                success     INTEGER,
                ts          DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def record_task(trace: dict, session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO task_runs (session_id, intent, duration_ms, success) VALUES (?, ?, ?, ?)",
            (session_id, str(trace.get("intent")), trace.get("duration_ms"), 1)
        )
        await db.commit()


async def record_failure(error: str, session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO task_runs (session_id, duration_ms, success) VALUES (?, ?, ?)",
            (session_id, 0, 0)
        )
        await db.commit()


async def main():
    await init_db()
    async with SumoKernel(SumoSettings(
        provider="ollama",
        model="phi3:mini",
        hooks={
            "on_task_complete": record_task,
            "on_task_failed": record_failure,
        },
    )) as kernel:
        trace = await kernel.run("Add error handling to database.py")
        print(trace.final_answer)

    # Show recent history
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT session_id, intent, duration_ms, success FROM task_runs ORDER BY ts DESC LIMIT 5") as cursor:
            rows = await cursor.fetchall()
            print("\n--- Recent Runs ---")
            for row in rows:
                status = "✓" if row[3] else "✗"
                print(f"  {status} {row[0][:8]}  intent={row[2]}  {row[2]}ms")


asyncio.run(main())
```

---

### Example 4 — Auto-Rollback on Failure

Automatically roll back all file changes when `on_task_failed` fires.

```python title="hooks/auto_rollback.py"
import asyncio
import subprocess
from sumospace import SumoKernel, SumoSettings

# Store the session_id → rollback needed mapping
_sessions_to_rollback: set[str] = set()


async def track_tools(tool: str, args: dict, step: int, session_id: str) -> None:
    """Mark this session as having made tool calls (needs potential rollback)."""
    if tool in {"write_file", "patch_file", "delete_file"}:
        _sessions_to_rollback.add(session_id)


async def auto_rollback(error: str, session_id: str) -> None:
    """Automatically invoke `sumo rollback` on failure."""
    if session_id not in _sessions_to_rollback:
        return  # No writes were made, nothing to roll back

    print(f"\n[AutoRollback] Task failed: {error}")
    print(f"[AutoRollback] Rolling back session {session_id}...")

    result = subprocess.run(
        ["sumo", "rollback", "--session", session_id, "--yes"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("[AutoRollback] Rollback succeeded.")
    else:
        print(f"[AutoRollback] Rollback failed:\n{result.stderr}")

    _sessions_to_rollback.discard(session_id)


async def main():
    async with SumoKernel(SumoSettings(
        provider="ollama",
        model="phi3:mini",
        hooks={
            "on_tool_call": track_tools,
            "on_task_failed": auto_rollback,
        },
    )) as kernel:
        trace = await kernel.run("Refactor utils.py to use dataclasses")
        print(trace.final_answer)


asyncio.run(main())
```
