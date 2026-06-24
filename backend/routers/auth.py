"""Authentication router — /auth/* endpoints."""

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from backend.database.db import (
    db_create_session,
    db_create_user,
    db_delete_session,
    db_get_conversation,
    db_get_consultations,
    db_get_conversations,
    db_get_user_by_email,
    db_get_user_by_username,
    db_update_user_voice,
)
from backend.dependencies.auth import get_current_user
from backend.schemas.auth import (
    ConsultHistoryItem,
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
        db_create_user(body.username, str(body.email), pwd_hash, salt)
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
    history = [
        ConsultHistoryItem(
            id=row["id"],
            transcript=row["transcript"],
            guidance=row["guidance"],
            escalate=bool(row["escalate"]),
            created_at=row["created_at"],
            conversation_id=row.get("conversation_id"),
        )
        for row in raw_history
    ]
    raw_convos = db_get_conversations(current_user["id"])
    conversations = [
        ConversationSummary(
            conversation_id=row["conversation_id"],
            started_at=row["started_at"],
            last_at=row["last_at"],
            turn_count=row["turn_count"],
            first_transcript=row["first_transcript"] or "",
        )
        for row in raw_convos
    ]
    return UserOut(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user.get("email", ""),
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
    return [
        ConsultHistoryItem(
            id=row["id"],
            transcript=row["transcript"],
            guidance=row["guidance"],
            escalate=bool(row["escalate"]),
            created_at=row["created_at"],
            conversation_id=row.get("conversation_id"),
        )
        for row in rows
    ]
