import json
import re
import tempfile
import os
from pathlib import Path
from typing import Any

try:
    import json_repair
except ImportError:
    json_repair = None


def _clean_json(raw: str) -> str:
    """Strip markdown fences and leading/trailing noise before parsing."""
    raw = raw.strip()
    
    # Strip markdown fences first
    if "```" in raw:
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = raw.replace("```", "")
    
    # Find the JSON object boundaries
    start = raw.find("{")
    if start == -1:
        return raw
    
    # Find the matching closing brace
    depth = 0
    end = start
    for i, char in enumerate(raw[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
                
    if depth != 0:
        end = raw.rfind("}")
        
    return raw[start:end+1] if end != -1 else raw[start:]


def parse_llm_json(raw: str, expected_keys: list[str] = None) -> tuple[dict[str, Any], bool]:
    """
    Parse a JSON string from an LLM.
    Uses json-repair as a fallback if strict parsing fails.
    
    Returns:
        (parsed_dict, repair_used)
    
    Raises:
        ValueError: If parsing fails entirely or if schema validation fails.
    """
    raw_clean = _clean_json(raw)
    
    # 1. Strict parse
    try:
        data = json.loads(raw_clean)
        _validate_schema(data, expected_keys)
        return data, False
    except json.JSONDecodeError as e:
        # If no json_repair available, fail
        if json_repair is None:
            raise ValueError(f"Strict parse failed and json-repair not installed: {e}") from e
            
    # 2. Tolerant parse via json-repair
    try:
        repaired_str = json_repair.repair_json(raw_clean, return_objects=False)
        # Parse the repaired string
        data = json.loads(repaired_str)
        
        # 3. Schema validation
        _validate_schema(data, expected_keys)
        
        # 4. Log the repair
        _log_repair(raw_clean, repaired_str)
        
        return data, True
    except Exception as e:
        raise ValueError(f"Tolerant JSON parsing failed: {e}") from e


def _validate_schema(data: Any, expected_keys: list[str] = None):
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dictionary.")
        
    if expected_keys:
        missing = [k for k in expected_keys if k not in data]
        if missing:
            raise ValueError(f"Parsed JSON missing expected keys: {missing}")


def _log_repair(original: str, repaired: str):
    """Log repaired JSON strings for telemetry/debugging."""
    log_file = Path(tempfile.gettempdir()) / "sumospace_parser_repairs.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ORIGINAL UNPARSABLE JSON:\n")
            f.write(original + "\n")
            f.write("-" * 80 + "\n")
            f.write("REPAIRED JSON:\n")
            f.write(repaired + "\n")
            f.write("=" * 80 + "\n\n")
    except Exception:
        pass
