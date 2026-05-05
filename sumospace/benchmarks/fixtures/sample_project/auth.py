"""auth.py — sample project fixture for benchmarks.
Missing docstrings, sync functions to convert, dead code included.
"""
import hashlib
import os
import time


# Dead code — unused variable
_LEGACY_SECRET = "hunter2"


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password, stored_hash):
    parts = stored_hash.split(":")
    if len(parts) != 2:
        return False
    salt, expected = parts
    _, actual_hash = hash_password(password, salt).split(":")
    return actual_hash == expected


def create_session(user_id):
    token = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()
    return {"user_id": user_id, "token": token, "created_at": time.time()}


def validate_session(session):
    if not session:
        return False
    age = time.time() - session.get("created_at", 0)
    return age < 3600


# Dead function — never called anywhere
def _old_validate(token):
    return len(token) == 64
