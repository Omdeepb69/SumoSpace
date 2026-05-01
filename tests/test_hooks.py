# tests/test_hooks.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sumospace.hooks import HookRegistry, HOOK_EVENTS


class TestHookRegistration:
    def test_register_valid_event(self):
        hooks = HookRegistry()
        fn = lambda: None
        hooks.register("on_task_start", fn)
        assert hooks.count("on_task_start") == 1

    def test_register_invalid_event_raises(self):
        hooks = HookRegistry()
        with pytest.raises(ValueError, match="Unknown event"):
            hooks.register("on_invalid_event", lambda: None)

    def test_decorator_registration(self):
        hooks = HookRegistry()

        @hooks.on("on_task_start")
        async def my_hook(task, session_id):
            pass

        assert hooks.count("on_task_start") == 1

    def test_decorator_invalid_event_raises(self):
        hooks = HookRegistry()
        with pytest.raises(ValueError, match="Unknown event"):
            @hooks.on("bogus_event")
            def bad_hook():
                pass

    def test_clear_specific_event(self):
        hooks = HookRegistry()
        hooks.register("on_task_start", lambda: None)
        hooks.register("on_task_complete", lambda: None)
        hooks.clear("on_task_start")
        assert hooks.count("on_task_start") == 0
        assert hooks.count("on_task_complete") == 1

    def test_clear_all(self):
        hooks = HookRegistry()
        hooks.register("on_task_start", lambda: None)
        hooks.register("on_task_complete", lambda: None)
        hooks.clear()
        assert hooks.count() == 0

    def test_registered_events(self):
        hooks = HookRegistry()
        hooks.register("on_task_start", lambda: None)
        hooks.register("on_step_complete", lambda: None)
        assert set(hooks.registered_events) == {"on_step_complete", "on_task_start"}

    def test_count_total(self):
        hooks = HookRegistry()
        hooks.register("on_task_start", lambda: None)
        hooks.register("on_task_start", lambda: None)
        hooks.register("on_task_complete", lambda: None)
        assert hooks.count() == 3


@pytest.mark.asyncio
class TestHookTrigger:
    async def test_async_hook_called(self):
        hooks = HookRegistry()
        called = []

        @hooks.on("on_task_start")
        async def my_hook(task, session_id):
            called.append((task, session_id))

        await hooks.trigger("on_task_start", "fix tests", "abc123")
        assert called == [("fix tests", "abc123")]

    async def test_sync_hook_called_via_executor(self):
        hooks = HookRegistry()
        called = []

        @hooks.on("on_task_start")
        def sync_hook(task, session_id):
            called.append((task, session_id))

        await hooks.trigger("on_task_start", "fix tests", "abc123")
        assert called == [("fix tests", "abc123")]

    async def test_multiple_hooks_all_called(self):
        hooks = HookRegistry()
        results = []

        @hooks.on("on_task_complete")
        async def hook_a(trace):
            results.append("a")

        @hooks.on("on_task_complete")
        async def hook_b(trace):
            results.append("b")

        await hooks.trigger("on_task_complete", MagicMock())
        assert results == ["a", "b"]

    async def test_hook_exception_does_not_propagate(self):
        hooks = HookRegistry(verbose=False)

        @hooks.on("on_task_start")
        async def broken_hook(task, session_id):
            raise RuntimeError("This hook is broken")

        # Should complete without raising
        await hooks.trigger("on_task_start", "task", "sid")

    async def test_hook_exception_does_not_affect_other_hooks(self):
        hooks = HookRegistry(verbose=False)
        called = []

        @hooks.on("on_task_start")
        async def broken_hook(task, session_id):
            raise RuntimeError("broken")

        @hooks.on("on_task_start")
        async def good_hook(task, session_id):
            called.append("good")

        await hooks.trigger("on_task_start", "task", "sid")
        assert called == ["good"]

    async def test_trigger_no_hooks_is_noop(self):
        hooks = HookRegistry()
        # Should not raise
        await hooks.trigger("on_task_start", "task", "sid")

    async def test_all_hook_events_are_valid(self):
        """Verify HOOK_EVENTS contains all expected lifecycle events."""
        expected = {
            "on_kernel_boot", "on_kernel_shutdown",
            "on_task_start", "on_plan_approved", "on_plan_rejected",
            "on_step_start", "on_step_complete", "on_step_failed",
            "on_task_complete", "on_task_failed",
        }
        assert HOOK_EVENTS == expected
