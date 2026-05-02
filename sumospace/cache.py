import hashlib
import json
import time
from pathlib import Path

from sumospace.committee import ExecutionPlan, ExecutionStep

class PlanCache:
    """
    Content-addressed cache for approved ExecutionPlans.
    Skips the 3-LLM committee on repeat tasks in the same context.
    """

    def __init__(self, cache_dir: str = ".sumo_db/plan_cache", ttl_hours: float = 24.0):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_hours * 3600

    def _key(self, task: str, context: str) -> str:
        raw = f"{task}|||{context[:500]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, task: str, context: str) -> ExecutionPlan | None:
        path = self._dir / f"{self._key(task, context)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if time.time() - data["cached_at"] > self._ttl:
                path.unlink()
                return None
            return self._deserialize(data["plan"])
        except Exception:
            return None

    def set(self, task: str, context: str, plan: ExecutionPlan):
        path = self._dir / f"{self._key(task, context)}.json"
        path.write_text(json.dumps({
            "cached_at": time.time(),
            "task": task,
            "plan": self._serialize(plan),
        }, indent=2))

    def _serialize(self, plan: ExecutionPlan) -> dict:
        return {
            "task": plan.task,
            "reasoning": plan.reasoning,
            "estimated_duration_s": plan.estimated_duration_s,
            "risks": plan.risks,
            "steps": [
                {
                    "step_number": s.step_number,
                    "tool": s.tool,
                    "description": s.description,
                    "parameters": s.parameters,
                    "expected_output": s.expected_output,
                    "critical": s.critical,
                }
                for s in plan.steps
            ],
        }

    def _deserialize(self, data: dict) -> ExecutionPlan:
        return ExecutionPlan(
            task=data["task"],
            steps=[ExecutionStep(**s) for s in data["steps"]],
            reasoning=data.get("reasoning", ""),
            estimated_duration_s=data.get("estimated_duration_s", 0),
            risks=data.get("risks", []),
            approved=True,
            approval_notes="Restored from cache",
        )

    def invalidate(self, task: str, context: str):
        path = self._dir / f"{self._key(task, context)}.json"
        if path.exists():
            path.unlink()

    def clear(self):
        for p in self._dir.glob("*.json"):
            p.unlink()

    def stats(self) -> dict:
        entries = list(self._dir.glob("*.json"))
        return {"count": len(entries), "size_mb": sum(p.stat().st_size for p in entries) / 1e6}
