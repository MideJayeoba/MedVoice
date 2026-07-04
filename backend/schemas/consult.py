"""Pydantic schemas for voice consultation endpoints."""

from pydantic import BaseModel, Field


class ReasonRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ReasonResponse(BaseModel):
    guidance: str
    escalate: bool
    normalized_query: str | None = None
    contexts_used: list[str] = []


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TranscribeResponse(BaseModel):
    transcript: str


class TriageResult(BaseModel):
    category: str
    category_confidence: float
    department: str
    department_confidence: float
    priority: str
    priority_confidence: float
    confidence: float
    confidence_band: str  # high | medium | low


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None


class TriagePredictRequest(BaseModel):
    conversation_id: str | None = None
    # Optional not-yet-sent text to include (e.g. what's typed in the box)
    message: str | None = None


class TriagePredictResponse(BaseModel):
    triage: TriageResult | None = None
    detail: str | None = None   # human message when nothing could be predicted


class ChatResponse(BaseModel):
    reply: str
    triage: TriageResult | None = None
    escalate: bool = False
    conversation_id: str | None = None
