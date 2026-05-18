import pytest
import os
import tempfile
from pathlib import Path
from sumospace.parsing import parse_llm_json, _log_repair


def test_strict_parse_success():
    raw = '{"tool": "read_file", "steps": 1}'
    data, repaired = parse_llm_json(raw, expected_keys=["tool"])
    assert repaired is False
    assert data["tool"] == "read_file"


def test_tolerant_parse_unescaped_newlines():
    # Llama3 classic failure: raw newlines inside strings
    raw = '''{
        "steps": 1,
        "parameters": {
            "old_text": "def foo():
    pass"
        }
    }'''
    data, repaired = parse_llm_json(raw, expected_keys=["steps"])
    assert repaired is True
    assert "def foo():\n    pass" in data["parameters"]["old_text"]


def test_tolerant_parse_trailing_commas():
    raw = '{"tool": "shell", "steps": [{"a": 1},],}'
    data, repaired = parse_llm_json(raw, expected_keys=["tool"])
    assert repaired is True
    assert data["tool"] == "shell"
    assert data["steps"][0]["a"] == 1


def test_tolerant_parse_markdown_fences():
    raw = '''```json
    {"tool": "shell", "steps": []}
    ```'''
    data, repaired = parse_llm_json(raw, expected_keys=["tool", "steps"])
    # Markdown stripping is part of _clean_json, so it shouldn't count as "repaired" via json-repair
    assert repaired is False
    assert data["tool"] == "shell"


def test_tolerant_parse_unescaped_quotes():
    raw = '{"tool": "shell", "parameters": {"cmd": "echo "hello world""}, "steps": []}'
    data, repaired = parse_llm_json(raw, expected_keys=["tool"])
    assert repaired is True
    assert "hello world" in data["parameters"]["cmd"]


def test_irreparable_garbage():
    raw = 'Sure, I can help with that. Here is the plan: I will read the file.'
    with pytest.raises(ValueError, match="Tolerant JSON parsing failed"):
        parse_llm_json(raw, expected_keys=["tool"])


def test_schema_validation_failure():
    raw = '{"tool": "shell"}'
    with pytest.raises(ValueError, match="Parsed JSON missing expected keys"):
        parse_llm_json(raw, expected_keys=["steps"])


def test_logging_writes_to_file():
    log_file = Path(tempfile.gettempdir()) / "sumospace_parser_repairs.log"
    if log_file.exists():
        log_file.unlink()

    # Trigger a repair
    raw = '{"steps": 1, "tool": "test",}'
    parse_llm_json(raw)

    assert log_file.exists()
    content = log_file.read_text()
    assert "ORIGINAL UNPARSABLE JSON" in content
    assert "REPAIRED JSON" in content
