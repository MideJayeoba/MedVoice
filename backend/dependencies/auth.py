"""FastAPI dependency injection — JWT authentication guards.

Flow:
  1. Decode + verify the JWT signature and expiry claim (stateless).
  2. Check the jti is still in the sessions table (allowlist) so logout works.
  3. Return the full user row from the DB.

Use `Depends(get_current_user)` on protected endpoints.
Use `Depends(get_optional_user)` where auth is optional.
"""

import logging
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.database.db import db_get_session, db_get_user_by_id
from backend.services.auth import decode_access_token

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

_UNAUTH = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token — please log in again",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Verify JWT — raises 401 if missing, invalid, expired, or logged out."""
    if credentials is None:
        raise _UNAUTH
    user = _resolve(credentials.credentials, strict=True)
    if user is None:
        raise _UNAUTH
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict | None:
    """Resolve JWT — returns None instead of raising for missing/invalid tokens."""
    if credentials is None:
        return None
    return _resolve(credentials.credentials, strict=False)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _resolve(token: str, strict: bool) -> dict | None:
    # 1. Verify signature + expiry
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        if strict:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired — please log in again",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid JWT: %s", exc)
        if strict:
            raise _UNAUTH
        return None

    # 2. Check the jti is in the DB (allowlist — removed on logout)
    jti = payload.get("jti")
    if jti and not db_get_session(jti):
        logger.debug("JWT jti not in sessions (logged out)")
        if strict:
            raise _UNAUTH
        return None

    # 3. Load full user row
    user_id = int(payload["sub"])
    user = db_get_user_by_id(user_id)
    if not user:
        if strict:
            raise _UNAUTH
        return None

    return user
