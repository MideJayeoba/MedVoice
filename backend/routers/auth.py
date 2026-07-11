"""Authentication router — /auth/* endpoints."""

import logging
import os
import secrets
import sqlite3

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from backend.database.db import (
    db_create_session,
    db_create_user,
    db_delete_conversation,
    db_delete_session,
    db_delete_user,
    db_get_conversation,
    db_get_consultations,
    db_get_conversations,
    db_get_user_by_email,
    db_get_user_by_username,
    db_update_user_voice,
)
from backend.dependencies.auth import get_current_user
from backend.dependencies.ratelimit import login_limiter, register_limiter
from backend.schemas.auth import (
    ConsultHistoryItem,
    GoogleLogin,
    VoiceUpdate,
    ConversationSummary,
    TokenResponse,
    UserLogin,
    UserOut,
    UserRegister,
)
from backend.services.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    session_expiry,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    dependencies=[Depends(register_limiter)],
)
def register(body: UserRegister) -> dict:
    # Check username and email uniqueness separately for clear error messages
    if db_get_user_by_username(body.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken — please choose another",
        )
    if db_get_user_by_email(body.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists",
        )
    try:
        pwd_hash, salt = hash_password(body.password)
        db_create_user(
            body.username, str(body.email), pwd_hash, salt,
            first_name=body.first_name.strip(),
            middle_name=body.middle_name.strip() if body.middle_name else None,
            last_name=body.last_name.strip(),
            birthdate=body.birthdate,
            gender=body.gender,
        )
        logger.info("New user registered: %s <%s>", body.username, body.email)
        return {"status": "registered", "username": body.username}
    except sqlite3.IntegrityError as exc:
        detail = "Username or email already taken"
        if "username" in str(exc).lower():
            detail = "Username already taken — please choose another"
        elif "email" in str(exc).lower():
            detail = "An account with this email already exists"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with username or email + password",
    dependencies=[Depends(login_limiter)],
)
def login(body: UserLogin) -> TokenResponse:
    # Try identifier as email first, then as username
    identifier = body.identifier.strip()
    user = None
    if "@" in identifier:
        user = db_get_user_by_email(identifier)
    if not user:
        user = db_get_user_by_username(identifier)

    if not user or not verify_password(body.password, user["password_hash"], user["salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user["id"], user["username"])
    # Store the jti in the sessions table so we can invalidate on logout
    payload = decode_access_token(token)
    db_create_session(payload["jti"], user["id"], session_expiry())
    logger.info("User logged in: %s", user["username"])
    return TokenResponse(access_token=token)


@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Login or register with a Google ID token",
)
def google_login(body: GoogleLogin) -> TokenResponse:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured on this server",
        )

    # Verify the ID token with Google (checks signature, expiry, issuer)
    try:
        r = httpx.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": body.credential},
            timeout=10.0,
        )
    except Exception as exc:
        logger.exception("Google tokeninfo unreachable: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach Google — try again")

    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    info = r.json()

    # The token must have been issued for OUR client id
    if info.get("aud") != client_id:
        logger.warning("Google token aud mismatch: %s", info.get("aud"))
        raise HTTPException(status_code=401, detail="Google token was not issued for this app")
    if info.get("email_verified") not in ("true", True):
        raise HTTPException(status_code=401, detail="Google account email is not verified")

    email = info["email"].lower()
    user = db_get_user_by_email(email)

    if not user:
        # First Google sign-in: create an account. Password is a random
        # throwaway — these users authenticate via Google, not passwords.
        base = email.split("@")[0][:40] or "user"
        username = base
        while db_get_user_by_username(username):
            username = f"{base}{secrets.randbelow(10000)}"
        pwd_hash, salt = hash_password(secrets.token_urlsafe(32))
        db_create_user(
            username, email, pwd_hash, salt,
            first_name=info.get("given_name"),
            last_name=info.get("family_name"),
        )
        user = db_get_user_by_email(email)
        logger.info("New user via Google: %s <%s>", username, email)

    token = create_access_token(user["id"], user["username"])
    payload = decode_access_token(token)
    db_create_session(payload["jti"], user["id"], session_expiry())
    logger.info("Google login: %s", user["username"])
    return TokenResponse(access_token=token)


@router.post(
    "/logout-token",
    summary="Invalidate a specific session token",
    include_in_schema=False,
)
def logout_token(body: dict) -> dict:
    token = body.get("token", "")
    if token:
        try:
            payload = decode_access_token(token)
            db_delete_session(payload["jti"])
        except Exception:
            pass  # expired or invalid — already harmless
    return {"status": "logged_out"}


@router.get(
    "/me",
    response_model=UserOut,
    summary="Get current user profile and consultation history",
)
def me(current_user: dict = Depends(get_current_user)) -> UserOut:
    raw_history = db_get_consultations(current_user["id"])
    history = [_history_item(row) for row in raw_history]
    raw_convos = db_get_conversations(current_user["id"])
    conversations = [
        ConversationSummary(
            conversation_id=row["conversation_id"],
            started_at=row["started_at"],
            last_at=row["last_at"],
            turn_count=row["turn_count"],
            first_transcript=row["first_transcript"] or "",
            priority=row.get("priority"),
            department=row.get("department"),
            category=row.get("category"),
        )
        for row in raw_convos
    ]
    return UserOut(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user.get("email", ""),
        first_name=current_user.get("first_name"),
        middle_name=current_user.get("middle_name"),
        last_name=current_user.get("last_name"),
        birthdate=current_user.get("birthdate"),
        gender=current_user.get("gender"),
        created_at=current_user["created_at"],
        tts_voice=current_user.get("tts_voice", "Ezinne"),
        history=history,
        conversations=conversations,
    )


@router.post(
    "/voice",
    summary="Update the current user's preferred TTS voice",
)
def update_voice(
    body: VoiceUpdate,
    current_user: dict = Depends(get_current_user),
) -> dict:
    allowed = {"Ezinne", "Abeo"}
    if body.voice not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Voice must be one of: {', '.join(allowed)}",
        )
    db_update_user_voice(current_user["id"], body.voice)
    logger.info("User %s switched voice to %s", current_user["username"], body.voice)
    return {"voice": body.voice}


@router.get(
    "/conversations/{conversation_id}",
    summary="Get all turns for a specific conversation",
)
def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
) -> list[ConsultHistoryItem]:
    rows = db_get_conversation(conversation_id)
    # Security: only return if the conversation belongs to this user
    for row in rows:
        if row["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Not your conversation")
    return [_history_item(row) for row in rows]


@router.get(
    "/export",
    summary="Download all personal data (NDPR right of access / portability)",
)
def export_my_data(current_user: dict = Depends(get_current_user)) -> dict:
    """Everything we hold about the user, in portable JSON."""
    consults = db_get_consultations(current_user["id"])
    return {
        "profile": {
            k: current_user.get(k)
            for k in ("username", "email", "first_name", "middle_name", "last_name",
                      "birthdate", "gender", "tts_voice", "created_at")
        },
        "consultations": [
            {k: r.get(k) for k in ("transcript", "guidance", "escalate", "conversation_id",
                                   "triage_category", "triage_department", "triage_priority",
                                   "triage_confidence", "created_at")}
            for r in consults
        ],
    }


@router.delete(
    "/me",
    summary="Permanently delete account and ALL associated data (NDPR right to erasure)",
)
def delete_my_account(current_user: dict = Depends(get_current_user)) -> dict:
    """Deletes the user row; sessions and consultations cascade with it."""
    db_delete_user(current_user["id"])
    logger.info("Account deleted (user id %s)", current_user["id"])
    return {"status": "deleted"}


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete a specific conversation and all its turns",
)
def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Erase every turn in a conversation owned by this user."""
    deleted = db_delete_conversation(conversation_id, current_user["id"])
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    logger.info("User %s deleted conversation %s (%d turns)", current_user["id"], conversation_id, deleted)
    return {"status": "deleted", "turns_removed": deleted}


def _history_item(row: dict) -> ConsultHistoryItem:
    return ConsultHistoryItem(
        id=row["id"],
        transcript=row["transcript"],
        guidance=row["guidance"],
        escalate=bool(row["escalate"]),
        created_at=row["created_at"],
        conversation_id=row.get("conversation_id"),
        triage_category=row.get("triage_category"),
        triage_department=row.get("triage_department"),
        triage_priority=row.get("triage_priority"),
        triage_confidence=row.get("triage_confidence"),
    )
