"""End-to-end voice consult pipeline (simplified)."""

import logging
import threading

from backend.services.asr import get_asr_status, transcribe_audio
from backend.services.llm import generate_guidance, get_reasoning_status
from backend.services.tts import get_tts_status, synthesize_speech

logger = logging.getLogger(__name__)


def get_system_status() -> dict:
    """Get status of system components."""
    return {
        "asr": get_asr_status(),
        "reasoning": get_reasoning_status(),
        "tts": get_tts_status(),
    }


def preload_models() -> None:
    """Background preload of ASR model."""
    def _asr():
        try:
            if get_asr_status()["mode"] == "whisper":
                from backend.services.asr import _load_whisper
                _load_whisper()
        except Exception as exc:
            logger.error("ASR preload failed: %s", exc)

    threading.Thread(target=_asr, daemon=True).start()


def run_voice_consult(
    audio_bytes: bytes,
    content_type: str | None,
    voice: str = "Ezinne",
    history: list[dict] | None = None,
) -> tuple[bytes, dict]:
    """Full pipeline: transcribe → direct LLM reasoning → TTS."""
    transcript = transcribe_audio(audio_bytes, content_type)
    guidance = generate_guidance(transcript, history=history)
    wav = synthesize_speech(guidance, voice=voice)

    meta = {
        "transcript": transcript,
        "normalized_query": None,
        "escalate": False,
        "guidance_length": len(guidance),
        "guidance_preview": guidance[:160],
        "guidance": guidance,
    }
    logger.info("Pipeline complete: %s", meta)
    return wav, meta
