import time
from sumospace.cache import PlanCache
from sumospace.committee import ExecutionPlan, ExecutionStep

def test_plan_cache_set_and_get(tmp_path):
    cache = PlanCache(cache_dir=str(tmp_path), ttl_hours=1.0)
    
    plan = ExecutionPlan(
        task="Test task",
        steps=[ExecutionStep(step_number=1, tool="shell", description="echo hi", parameters={"cmd": "echo hi"}, expected_output="hi", critical=True)],
        reasoning="Because",
        estimated_duration_s=5,
        risks=[],
        approved=True,
        approval_notes=""
    )
    
    # Not in cache yet
    assert cache.get("Test task", "context") is None
    
    # Add to cache
    cache.set("Test task", "context", plan)
    
    # Retrieve from cache
    cached = cache.get("Test task", "context")
    assert cached is not None
    assert cached.task == "Test task"
    assert len(cached.steps) == 1
    assert cached.steps[0].tool == "shell"

def test_plan_cache_ttl_eviction(tmp_path):
    cache = PlanCache(cache_dir=str(tmp_path), ttl_hours=0.0001) # very short TTL
    
    plan = ExecutionPlan(
        task="Test task",
        steps=[],
        reasoning="",
        estimated_duration_s=0,
        risks=[],
        approved=True,
        approval_notes=""
    )
    
    cache.set("Test task", "context", plan)
    assert cache.get("Test task", "context") is not None
    
    # Wait for TTL to expire
    time.sleep(0.5)
    
    assert cache.get("Test task", "context") is None

def test_plan_cache_invalidation(tmp_path):
    cache = PlanCache(cache_dir=str(tmp_path), ttl_hours=1.0)
    
    plan = ExecutionPlan(
        task="Test task",
        steps=[],
        reasoning="",
        estimated_duration_s=0,
        risks=[],
        approved=True,
        approval_notes=""
    )
    
    cache.set("Test task", "context", plan)
    cache.invalidate("Test task", "context")
    assert cache.get("Test task", "context") is None
