# Sample Project

A minimal Python web service demonstrating authentication, database access, and utilities.

## Modules

- `auth.py` — Password hashing and session management
- `database.py` — SQLite CRUD operations
- `utils.py` — Text utilities and JSON helpers

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from auth import hash_password, verify_password, create_session
from database import get_user, create_user

pw_hash = hash_password("mysecret")
assert verify_password("mysecret", pw_hash)

session = create_session(user_id=42)
```
