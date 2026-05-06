# sumospace/benchmarks/fixtures/sample_project/dead_code.py
# BENCHMARK FIXTURE — do not modify
# Contains intentional dead code for the agent to remove

import os
import sys
import json          # used
import hashlib       # DEAD — never used
import base64        # DEAD — never used
from pathlib import Path  # used

# DEAD FUNCTION — never called anywhere
def _legacy_hash_password(password):
    import md5
    return md5.new(password).hexdigest()

# DEAD FUNCTION — never called anywhere  
def _old_format_date(date_string):
    parts = date_string.split("-")
    return f"{parts[2]}/{parts[1]}/{parts[0]}"

# DEAD FUNCTION — never called anywhere
def _deprecated_validate(value):
    if value == None:
        return False
    return True

# LIVE FUNCTION — keep this
def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

# Wait — hashlib IS used above. Remove base64 and the dead functions.

# LIVE FUNCTION — keep this
def read_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

# LIVE FUNCTION — keep this
def write_config(config_path: str, data: dict) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# DEAD VARIABLE — assigned but never used
_LEGACY_VERSION = "0.0.1-alpha"
_DEPRECATED_FLAG = True
