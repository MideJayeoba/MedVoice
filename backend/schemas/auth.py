"""Pydantic schemas for authentication endpoints."""

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """Login accepts either username or email in the `identifier` field."""
    identifier: str = Field(..., description="Username or email address")
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ConsultHistoryItem(BaseModel):
    id: int
    transcript: str
    guidance: str
    escalate: bool
    created_at: str
    conversation_id: str | None = None
    triage_category: str | None = None
    triage_department: str | None = None
    triage_priority: str | None = None
    triage_confidence: float | None = None


class ConversationSummary(BaseModel):
    conversation_id: str
    started_at: str
    last_at: str
    turn_count: int
    first_transcript: str
    priority: str | None = None      # session-level urgency (highest reached)
    department: str | None = None    # most recent suggested department
    category: str | None = None      # most recent condition area


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    tts_voice: str = "Ezinne"
    created_at: str
    history: list[ConsultHistoryItem] = []
    conversations: list[ConversationSummary] = []


class VoiceUpdate(BaseModel):
    voice: str


