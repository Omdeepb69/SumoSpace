from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import builtins
from filelock import FileLock

if TYPE_CHECKING:
    from sumospace.kernel import ExecutionTrace
from sumospace.committee import CommitteeVerdict
from sumospace.settings import SumoSettings

class AuditLogger:
    """
    Logs execution traces and committee verdicts to an audit log file.
    Used for observability, debugging, and auditing.
    """
    def __init__(self, settings: SumoSettings):
        self.settings = settings
        self.log_dir = Path(settings.workspace) / ".sumo_audit"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
    def log(self, trace: ExecutionTrace, verdict: Optional[CommitteeVerdict] = None):
        """Write the trace and verdict to a JSON line in the audit log."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": trace.session_id,
            "task": trace.task,
            "success": trace.success,
            "duration_ms": trace.duration_ms,
            "intent": trace.intent.name if trace.intent else "UNKNOWN",
            "final_answer": trace.final_answer,
            "error": trace.error,
        }
        
        if verdict:
            log_entry["committee_verdict"] = {
                "approved": verdict.approved,
                "rejection_reason": verdict.rejection_reason,
                "critic_output": verdict.critic_output,
                "resolver_output": verdict.resolver_output,
                "planner_output": verdict.planner_output,
            }
            
        log_entry["steps"] = [
            {
                "step_number": st.step_number,
                "tool": st.tool,
                "description": st.description,
                "success": st.result.success,
                "duration_ms": st.duration_ms,
                "error": st.result.error,
            }
            for st in trace.step_traces
        ]

        log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            self._update_index(log_entry)
        except Exception:
            pass

    def _update_index(self, entry: dict):
        """Incrementally update stats_index.json."""
        index_file = self.log_dir / "stats_index.json"
        lock_file = self.log_dir / "stats_index.json.lock"
        
        try:
            with FileLock(lock_file, timeout=10):
                if index_file.exists():
                    with open(index_file, "r", encoding="utf-8") as f:
                        stats = json.load(f)
                else:
                    stats = {
                        "total_sessions": 0,
                        "successful_sessions": 0,
                        "failed_sessions": 0,
                        "total_duration_ms": 0.0,
                        "tool_usage": {},
                        "intent_usage": {},
                        "failure_reasons": {},
                    }

                stats["total_sessions"] += 1
                if entry["success"]:
                    stats["successful_sessions"] += 1
                else:
                    stats["failed_sessions"] += 1
                    reason = entry.get("error") or "Unknown error"
                    stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + 1

                stats["total_duration_ms"] += entry["duration_ms"]
                
                intent = entry["intent"]
                stats["intent_usage"][intent] = stats["intent_usage"].get(intent, 0) + 1

                for step in entry.get("steps", []):
                    tool = step["tool"]
                    if tool not in stats["tool_usage"]:
                        stats["tool_usage"][tool] = {"success": 0, "fail": 0}
                    if step["success"]:
                        stats["tool_usage"][tool]["success"] += 1
                    else:
                        stats["tool_usage"][tool]["fail"] += 1

                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
        except Exception:
            pass

    def list(self, limit: int = 20) -> builtins.list[dict]:
        """
        List recent execution sessions from all log files.

        Note:
            This scans the `.jsonl` files in reverse chronological order. Use it 
            for displaying a history dashboard.

        Warning:
            The returned dictionaries contain full trace data which can be memory 
            intensive if `limit` is set very high.
        """
        sessions = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)
        for log_file in log_files:
            if len(sessions) >= limit:
                break
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        sessions.append(json.loads(line))
                        if len(sessions) >= limit:
                            break
            except Exception:
                continue
        return sessions

    def show(self, session_id: str) -> Optional[dict]:
        """
        Retrieve the full trace for a specific session.

        Note:
            This scans the log files to find the exact session. Use this when you need
            to inspect the exact steps, tool outputs, and LLM reasoning.

        Warning:
            Returns `None` if the session ID does not exist. Always check for `None`
            before attempting to parse the result.
        """
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry["session_id"] == session_id:
                            return entry
            except Exception:
                continue
        return None

    def search(self, query: str, limit: int = 10) -> builtins.list[dict]:
        """
        Search sessions for a substring in the task.

        Note:
            This is currently a linear substring search over the log files. It is useful 
            for debugging (e.g. `audit.search("database")`).

        Warning:
            Because it uses substring matching, it is not semantic. Searching for
            "DB" will not match tasks that only say "database".
        """
        results = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)
        for log_file in log_files:
            if len(results) >= limit:
                break
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in reversed(f.readlines()):
                        entry = json.loads(line)
                        if query.lower() in entry["task"].lower():
                            results.append(entry)
                            if len(results) >= limit:
                                break
            except Exception:
                continue
        return results

    def stats(self) -> dict:
        """
        Get aggregated stats from the index.

        Note:
            These statistics are maintained incrementally in `stats_index.json`. 
            Calling `stats()` is fast and perfectly safe to use in a high-frequency polling endpoint.
        """
        index_file = self.log_dir / "stats_index.json"
        if not index_file.exists():
            return {}
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def export(self, session_id: str) -> Optional[str]:
        """
        Export session to a Markdown report.

        Note:
            Useful for downloading logs via a web API or saving a specific
            run to a bug report attachment.
        """
        session = self.show(session_id)
        if not session:
            return None
        
        lines = [
            f"# Session Audit Report: {session_id}",
            f"- **Task**: {session['task']}",
            f"- **Timestamp**: {session['timestamp']}",
            f"- **Success**: {'✅' if session['success'] else '❌'}",
            f"- **Duration**: {session['duration_ms']:.0f}ms",
            f"- **Intent**: {session['intent']}",
            "",
            "## Committee Verdict",
        ]
        
        verdict = session.get("committee_verdict")
        if verdict:
            lines.append(f"- **Approved**: {verdict['approved']}")
            if not verdict["approved"]:
                lines.append(f"- **Rejection Reason**: {verdict['rejection_reason']}")
        else:
            lines.append("No committee deliberation (likely cached).")
            
        lines.extend([
            "",
            "## Execution Steps",
            "| # | Tool | Description | Success | Duration |",
            "|---|------|-------------|---------|----------|",
        ])
        
        for step in session.get("steps", []):
            status = "✅" if step["success"] else "❌"
            lines.append(
                f"| {step['step_number']} | `{step['tool']}` | {step['description']} | "
                f"{status} | {step['duration_ms']:.0f}ms |"
            )
            
        lines.extend([
            "",
            "## Final Answer",
            session.get("final_answer", "N/A"),
        ])
        
        if session.get("error"):
            lines.extend([
                "",
                "## Error Details",
                f"```\n{session['error']}\n```",
            ])
            
        return "\n".join(lines)
