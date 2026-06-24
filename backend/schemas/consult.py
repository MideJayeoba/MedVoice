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
