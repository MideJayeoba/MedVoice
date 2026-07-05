"""End-to-end voice consult pipeline (simplified)."""

import logging
import threading

from backend.services.asr import get_asr_status, transcribe_audio
from backend.services.llm import generate_guidance, get_reasoning_status
from backend.services.triage import (
    get_triage_status,
    has_emergency_signal,
    has_medical_signal,
    predict_triage,
    preload_triage,
)
from backend.services.tts import get_tts_status, synthesize_speech

logger = logging.getLogger(__name__)


def get_system_status() -> dict:
    """Get status of system components."""
    return {
        "asr": get_asr_status(),
        "reasoning": get_reasoning_status(),
        "tts": get_tts_status(),
        "triage": get_triage_status(),
    }


def preload_models() -> None:
    """Background preload of ASR + triage models."""
    def _asr():
        try:
            status = get_asr_status()
            if status["mode"] == "whisper" and not status.get("groq_available"):
                from backend.services.asr import _load_whisper
                _load_whisper()
        except Exception as exc:
            logger.error("ASR preload failed: %s", exc)

    threading.Thread(target=_asr, daemon=True).start()
    preload_triage()


def triage_for_conversation(
    query: str, history: list[dict] | None, force: bool = False
) -> dict | None:
    """Predict triage from ALL patient text in this conversation.

    Runs on every turn and re-checks the whole conversation, so the
    prediction keeps updating as new symptoms are mentioned. It stays quiet
    on non-health chat so a greeting never produces a wrong prediction that
    would pollute the LLM's reply.

    - force=True: predict regardless of the signal gate (the manual
      "Predict now" button). predict_triage still rejects truly off-topic text.
    - Emergencies (red-flag phrases) surface immediately, on any turn.
    - Otherwise predict as soon as the conversation carries medical signal
      (one clear symptom word is enough).
    """
    parts = [h["transcript"] for h in (history or []) if h.get("transcript")]
    if query:
        parts.append(query)
    combined = " . ".join(parts)

    if force or has_emergency_signal(query) or has_medical_signal(combined):
        return predict_triage(combined)

    return None


# Deterministic safety net: the LLM sometimes leads with first-aid even for
# an emergency, so we guarantee the urgent instruction is present ourselves.
_EMERGENCY_PREFIX = (
    "Please go to the nearest hospital now, or call for emergency help — "
    "this may be serious. "
)
_URGENT_MARKERS = ("hospital", "emergency", "ambulance", "call for help",
                   "call for emergency", "urgent", "right away", "immediately")


def _apply_emergency_safety(guidance: str, triage: dict | None) -> str:
    """Prepend a clear urgent instruction if triage says Emergency and the
    model didn't already tell the patient to seek urgent care."""
    if not triage or triage.get("priority") != "Emergency":
        return guidance
    if any(m in guidance.lower() for m in _URGENT_MARKERS):
        return guidance
    return _EMERGENCY_PREFIX + guidance


def run_consult_text(
    query: str,
    history: list[dict] | None = None,
    user_name: str | None = None,
) -> tuple[str, dict | None]:
    """Text-only pipeline: triage → LLM. Returns (guidance, triage)."""
    triage = triage_for_conversation(query, history)
    guidance = generate_guidance(query, history=history, triage=triage, user_name=user_name)
    guidance = _apply_emergency_safety(guidance, triage)
    return guidance, triage


def run_voice_consult(
    audio_bytes: bytes,
    content_type: str | None,
    voice: str = "Ezinne",
    history: list[dict] | None = None,
    user_name: str | None = None,
) -> tuple[bytes, dict]:
    """Full pipeline: transcribe → triage → LLM reasoning → TTS."""
    transcript = transcribe_audio(audio_bytes, content_type)
    triage = triage_for_conversation(transcript, history)
    guidance = generate_guidance(transcript, history=history, triage=triage, user_name=user_name)
    guidance = _apply_emergency_safety(guidance, triage)
    wav = synthesize_speech(guidance, voice=voice)

    meta = {
        "transcript": transcript,
        "normalized_query": None,
        "escalate": bool(triage and triage.get("priority") == "Emergency"),
        "guidance_length": len(guidance),
        "guidance_preview": guidance[:160],
        "guidance": guidance,
        "triage": triage,
    }
    logger.info("Pipeline complete: %s", {k: v for k, v in meta.items() if k != "guidance"})
    return wav, meta
