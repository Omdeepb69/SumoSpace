# sumospace/benchmarks/fixtures/sample_project/sync_io.py
# BENCHMARK FIXTURE — do not modify
# Contains sync I/O that should be converted to async

import time
import requests
from pathlib import Path


def fetch_user(user_id: int) -> dict:
    """Fetch user from API."""
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()


def fetch_posts(user_id: int) -> list:
    """Fetch posts for a user."""
    time.sleep(0.1)  # simulated delay
    response = requests.get(f"https://api.example.com/users/{user_id}/posts")
    return response.json()


def save_result(path: str, data: dict) -> None:
    """Save result to disk."""
    time.sleep(0.05)  # simulated delay
    Path(path).write_text(str(data))


def process_users(user_ids: list) -> list:
    """Process multiple users sequentially."""
    results = []
    for uid in user_ids:
        user = fetch_user(uid)
        posts = fetch_posts(uid)
        results.append({"user": user, "posts": posts})
    return results


def health_check(url: str) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False
