import json
import pytest
from pathlib import Path
from sumospace.audit import AuditLogger
from sumospace.settings import SumoSettings
from sumospace.kernel import ExecutionTrace, Intent, ClassificationResult, ToolResult, StepTrace

@pytest.fixture
def audit_logger(tmp_path):
    settings = SumoSettings(workspace=str(tmp_path))
    return AuditLogger(settings)

def test_audit_log_and_index(audit_logger):
    trace = ExecutionTrace(
        task="Test task",
        session_id="session-123",
        intent=Intent.WRITE_CODE,
        classification=ClassificationResult(
            intent=Intent.WRITE_CODE, 
            confidence=0.9, 
            needs_execution=True,
            needs_web=False,
            needs_retrieval=True,
            reasoning="test"
        ),
        plan=None,
    )
    trace.success = True
    trace.duration_ms = 1500.0
    trace.final_answer = "Done."
    trace.step_traces.append(StepTrace(
        step_number=1, tool="read_file", description="read", 
        result=ToolResult(tool="read_file", success=True, output="content"), duration_ms=200.0
    ))

    audit_logger.log(trace)

    # Check file exists
    log_files = list(audit_logger.log_dir.glob("audit_*.jsonl"))
    assert len(log_files) == 1
    
    # Check index exists
    index_file = audit_logger.log_dir / "stats_index.json"
    assert index_file.exists()
    
    with open(index_file, "r") as f:
        stats = json.load(f)
        assert stats["total_sessions"] == 1
        assert stats["successful_sessions"] == 1
        assert stats["tool_usage"]["read_file"]["success"] == 1
        assert stats["intent_usage"]["WRITE_CODE"] == 1

def test_list_sessions(audit_logger):
    for i in range(5):
        trace = ExecutionTrace(task=f"Task {i}", session_id=f"sid-{i}", intent=Intent.GENERAL_QA, classification=None, plan=None)
        trace.success = True
        audit_logger.log(trace)
        
    sessions = audit_logger.list(limit=3)
    assert len(sessions) == 3
    assert sessions[0]["task"] == "Task 4"  # Most recent first

def test_search_sessions(audit_logger):
    trace1 = ExecutionTrace(task="Fix auth.py", session_id="s1", intent=Intent.WRITE_CODE, classification=None, plan=None)
    trace2 = ExecutionTrace(task="Read docs", session_id="s2", intent=Intent.GENERAL_QA, classification=None, plan=None)
    audit_logger.log(trace1)
    audit_logger.log(trace2)
    
    results = audit_logger.search("auth")
    assert len(results) == 1
    assert results[0]["session_id"] == "s1"

def test_export_markdown(audit_logger):
    trace = ExecutionTrace(task="Export me", session_id="exp-1", intent=Intent.WRITE_CODE, classification=None, plan=None)
    trace.success = True
    trace.step_traces.append(StepTrace(1, "tool", "desc", ToolResult("tool", True, "out"), 100.0))
    audit_logger.log(trace)
    
    md = audit_logger.export("exp-1")
    assert "# Session Audit Report: exp-1" in md
    assert "Export me" in md
    assert "| 1 | `tool` | desc | ✅ | 100ms |" in md
