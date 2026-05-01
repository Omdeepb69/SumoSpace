# tests/test_templates.py

import pytest
from sumospace.templates import TemplateManager, SafeFormatMap, TEMPLATE_VARIABLES


class TestSafeFormatMap:
    def test_returns_value_for_existing_key(self):
        m = SafeFormatMap({"name": "Alice"})
        assert "Hello Alice".format_map(m) == "Hello Alice"

    def test_returns_empty_for_missing_key(self):
        m = SafeFormatMap({"name": "Alice"})
        result = "Hello {name}, your {role}".format_map(m)
        assert result == "Hello Alice, your "

    def test_handles_all_missing(self):
        m = SafeFormatMap({})
        result = "{a} {b} {c}".format_map(m)
        assert result == "  "


class TestTemplateManager:
    def test_defaults_loaded(self):
        tm = TemplateManager()
        assert "planner_prompt" in tm.available
        assert "critic_prompt" in tm.available
        assert "resolver_prompt" in tm.available
        assert "synthesis_prompt" in tm.available
        assert "system_prompt" in tm.available

    def test_get_renders_template(self):
        tm = TemplateManager()
        result = tm.get("system_prompt", version="0.1.0")
        assert "0.1.0" in result
        assert "Sumo" in result

    def test_get_missing_vars_returns_empty(self):
        tm = TemplateManager()
        result = tm.get("system_prompt")  # version not provided
        assert "Sumo" in result  # Still renders, just version is ""

    def test_get_unknown_template_returns_empty(self):
        tm = TemplateManager()
        result = tm.get("nonexistent_template", foo="bar")
        assert result == ""

    def test_raw_returns_unrendered(self):
        tm = TemplateManager()
        raw = tm.raw("system_prompt")
        assert raw is not None
        assert "{version}" in raw

    def test_raw_unknown_returns_none(self):
        tm = TemplateManager()
        assert tm.raw("nonexistent") is None

    def test_custom_template_overrides_default(self, tmp_path):
        custom = tmp_path / "planner_prompt.txt"
        custom.write_text("Custom planner: {task} with {tools}")

        tm = TemplateManager(template_path=str(tmp_path))
        result = tm.get("planner_prompt", task="fix tests", tools="shell, read_file")
        assert "Custom planner:" in result
        assert "fix tests" in result

    def test_custom_template_ignores_unknown_files(self, tmp_path):
        unknown = tmp_path / "totally_made_up.txt"
        unknown.write_text("This should be ignored")

        tm = TemplateManager(template_path=str(tmp_path))
        assert "totally_made_up" not in tm.available

    def test_validates_unknown_vars_at_load_time(self, tmp_path, capsys):
        custom = tmp_path / "planner_prompt.txt"
        custom.write_text("Task: {task}, Unknown: {foobar}")

        tm = TemplateManager(template_path=str(tmp_path))
        # Template still loads (warning is printed)
        result = tm.get("planner_prompt", task="test")
        assert "test" in result

    def test_nonexistent_path_is_skipped(self):
        tm = TemplateManager(template_path="/nonexistent/path/to/templates")
        # Falls back gracefully to defaults
        assert len(tm.available) == len(TEMPLATE_VARIABLES)

    def test_find_unknown_vars(self):
        unknown = TemplateManager._find_unknown_vars(
            "Hello {task}, your {bogus_var}", ["task", "context"]
        )
        assert unknown == ["bogus_var"]

    def test_find_unknown_vars_all_known(self):
        unknown = TemplateManager._find_unknown_vars(
            "Hello {task}", ["task", "context"]
        )
        assert unknown == []
