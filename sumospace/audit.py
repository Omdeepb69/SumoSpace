import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        except Exception as e:
            # Audit logging should never crash the main process
            pass
