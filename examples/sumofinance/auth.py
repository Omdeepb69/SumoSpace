"""
SumoFinance — Authentication & 2FA Module
==========================================
Bank-grade TOTP-based two-factor authentication using pyotp.
Zero cost (no SMS), works offline, authenticator-app compatible.

Flow:
  1. POST /auth/register  → creates user, returns QR code URI
  2. POST /auth/verify-2fa → verifies 6-digit OTP, returns JWT
  3. POST /auth/login      → verifies email+password, returns pending_2fa
  4. POST /auth/login/2fa  → verifies OTP after login, returns JWT
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

import pyotp
import jwt
from pydantic import BaseModel, EmailStr

# ─── Configuration ────────────────────────────────────────────────────────────

JWT_SECRET = os.environ.get("SUMOFINANCE_JWT_SECRET", "sumo-finance-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24
APP_NAME = "SumoFinance"
DB_PATH = "sumofinance.db"


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    phone_number: str = ""


class RegisterResponse(BaseModel):
    user_id: str
    message: str
    qr_code_uri: str


class VerifyOTPRequest(BaseModel):
    user_id: str
    otp_code: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    status: str
    user_id: str
    message: str


class RegisterResponse(BaseModel):
    user_id: str
    message: str
    qr_code_uri: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    expires_in_hours: int = JWT_EXPIRY_HOURS


# ─── Password Hashing (PBKDF2-SHA256, no extra deps) ─────────────────────────

def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash password with PBKDF2-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=100_000)
    return dk.hex(), salt.hex()


def _verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Verify a password against stored hash and salt."""
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(stored_salt), iterations=100_000
    )
    return hmac.compare_digest(dk.hex(), stored_hash)


# ─── JWT Token Generation ────────────────────────────────────────────────────

def _create_jwt(user_id: str, email: str) -> str:
    """Create a signed JWT access token."""
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> dict | None:
    """Verify and decode a JWT token. Returns payload or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


# ─── Database Schema ─────────────────────────────────────────────────────────

def init_auth_db():
    """Create the auth_users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS auth_users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            phone_number TEXT DEFAULT '',
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            totp_secret TEXT NOT NULL,
            totp_verified INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ─── Core Auth Functions ─────────────────────────────────────────────────────

def register_user(req: RegisterRequest) -> RegisterResponse:
    """
    Register a new user with TOTP 2FA.
    
    1. Hash the password with PBKDF2
    2. Generate a TOTP secret key
    3. Generate a QR code URI for authenticator apps
    4. Store everything in the database
    5. Return the QR URI (one-time only — never exposed again)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if email already exists
    c.execute("SELECT user_id FROM auth_users WHERE email = ?", (req.email,))
    if c.fetchone():
        conn.close()
        raise ValueError(f"Email '{req.email}' is already registered.")

    user_id = str(uuid.uuid4())
    password_hash, password_salt = _hash_password(req.password)

    # Generate TOTP secret
    totp_secret = pyotp.random_base32()
    totp = pyotp.TOTP(totp_secret)

    # Generate the QR code URI (otpauth:// format)
    qr_uri = totp.provisioning_uri(name=req.email, issuer_name=APP_NAME)

    c.execute(
        """INSERT INTO auth_users 
           (user_id, email, full_name, phone_number,
            password_hash, password_salt, totp_secret, totp_verified, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, req.email, req.full_name, req.phone_number,
            password_hash, password_salt,
            totp_secret, 0, datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    # Also create the user in the main finance DB so dashboard works
    import db
    try:
        # Link auth user to finance user with same ID
        finance_conn = sqlite3.connect(DB_PATH)
        fc = finance_conn.cursor()
        fc.execute(
            "INSERT OR IGNORE INTO users (user_id, name, email, created_at, current_balance) VALUES (?, ?, ?, ?, ?)",
            (user_id, req.full_name, req.email, datetime.now().isoformat(), 0.0),
        )
        finance_conn.commit()
        finance_conn.close()
    except Exception:
        pass  # Finance tables might not exist yet

    return RegisterResponse(
        user_id=user_id,
        message=f"Registration successful! Scan the QR code in your authenticator app (Google Authenticator, Authy, etc.).",
        qr_code_uri=qr_uri,
    )


def verify_otp(user_id: str, otp_code: str) -> TokenResponse:
    """
    Verify a 6-digit TOTP code and return a JWT access token.
    
    Uses a ±1 window (valid_window=1) to account for slight clock drift.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,))
    user = c.fetchone()
    if not user:
        conn.close()
        raise ValueError("User not found.")

    totp = pyotp.TOTP(user["totp_secret"])

    if not totp.verify(otp_code, valid_window=1):
        conn.close()
        raise ValueError("Invalid OTP code. Please check your authenticator app and try again.")

    # Mark TOTP as verified (first-time setup confirmation)
    if not user["totp_verified"]:
        c.execute("UPDATE auth_users SET totp_verified = 1 WHERE user_id = ?", (user_id,))
        conn.commit()

    conn.close()

    token = _create_jwt(user_id, user["email"])
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        expires_in_hours=JWT_EXPIRY_HOURS,
    )


def login_step1(email: str, password: str) -> LoginResponse:
    """
    Step 1 of login: verify email and password.
    
    Does NOT return a token — requires 2FA verification first.
    Returns a pending_2fa status with the user_id for Step 2.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM auth_users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()

    if not user:
        raise ValueError("Invalid email or password.")

    if not _verify_password(password, user["password_hash"], user["password_salt"]):
        raise ValueError("Invalid email or password.")

    return LoginResponse(
        status="pending_2fa",
        user_id=user["user_id"],
        message="Password verified. Please enter your 6-digit authenticator code to complete login.",
    )


def get_auth_user(user_id: str) -> dict | None:
    """Retrieve an auth user by ID (without sensitive fields)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT user_id, email, full_name, phone_number, totp_verified, created_at FROM auth_users WHERE user_id = ?",
        (user_id,),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


# Initialize auth tables on import
init_auth_db()
