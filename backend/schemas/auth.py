"""Pydantic schemas for authentication endpoints."""

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    first_name: str = Field(..., min_length=1, max_length=100)
    middle_name: str | None = Field(None, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    birthdate: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date YYYY-MM-DD")
    gender: Literal["Male", "Female", "Other"]


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
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    birthdate: str | None = None
    gender: str | None = None
    tts_voice: str = "Ezinne"
    created_at: str
    history: list[ConsultHistoryItem] = []
    conversations: list[ConversationSummary] = []


class VoiceUpdate(BaseModel):
    voice: str


class GoogleLogin(BaseModel):
    """ID token (credential) returned by Google Identity Services on the frontend."""
    credential: str = Field(..., min_length=20)


