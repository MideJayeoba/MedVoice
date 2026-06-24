"""Authentication business logic — password hashing and JWT token management."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from backend.config import JWT_ALGORITHM, JWT_SECRET, SESSION_TTL_HOURS


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def _derive_hash(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256, 260 000 iterations."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        260_000,
    )
    return dk.hex()


def hash_password(password: str) -> tuple[str, str]:
    """Return (password_hash, salt) for storage."""
    salt = secrets.token_hex(32)
    return _derive_hash(password, salt), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    return secrets.compare_digest(_derive_hash(password, salt), stored_hash)


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

def create_access_token(user_id: int, username: str) -> str:
    """Sign a JWT containing user identity and an expiry claim."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(hours=SESSION_TTL_HOURS),
        # jti lets us store this specific token in the DB for logout support
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT.
    Raises jwt.InvalidTokenError (or a subclass) on any failure.
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# Session helpers (kept for logout / DB allowlist)
# ---------------------------------------------------------------------------

def session_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)


def is_session_expired(expires_at_iso: str) -> bool:
    """Return True if the stored ISO timestamp is in the past."""
    try:
        exp = datetime.fromisoformat(expires_at_iso)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= exp
    except ValueError:
        return True
