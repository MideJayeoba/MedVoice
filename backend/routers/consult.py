"""Voice consultation router — /transcribe /reason /speak /consult."""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.database.db import db_get_conversation, db_save_consultation
from backend.dependencies.auth import get_optional_user
from backend.schemas.consult import ReasonRequest, ReasonResponse, SpeakRequest, TranscribeResponse
from backend.services.asr import transcribe_audio
from backend.services.llm import generate_guidance
from backend.services.pipeline import run_voice_consult
from backend.services.tts import synthesize_speech

logger = logging.getLogger(__name__)
router = APIRouter()

_ERROR_VOICE = (
    "Sorry oga, something no work for our side. "
    "Please try again small-small, or talk to the CHEW for help."
)


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Convert audio to text using Whisper ASR",
)
async def transcribe(
    audio: Annotated[UploadFile, File(...)],
) -> TranscribeResponse:
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        transcript = transcribe_audio(audio_bytes, audio.content_type)
        logger.info("Transcribed: %s", transcript[:80])
        return TranscribeResponse(transcript=transcript)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail="Transcription failed") from exc


@router.post(
    "/reason",
    response_model=ReasonResponse,
    summary="Text-based reasoning (no audio)",
)
def reason(
    body: ReasonRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
) -> ReasonResponse:
    try:
        guidance = generate_guidance(body.query)
        logger.info("Reasoned: %s", guidance[:120])

        if current_user:
            db_save_consultation(current_user["id"], body.query, guidance, False)

        return ReasonResponse(
            guidance=guidance,
            escalate=False,
            normalized_query=None,
            contexts_used=[],
        )
    except Exception as exc:
        logger.exception("Reasoning failed: %s", exc)
        raise HTTPException(status_code=500, detail="Reasoning failed") from exc


@router.post(
    "/speak",
    summary="Convert text to speech (WAV)",
    response_class=Response,
)
def speak(
    body: SpeakRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
) -> Response:
    try:
        voice = current_user.get("tts_voice", "Ezinne") if current_user else "Ezinne"
        wav_bytes = synthesize_speech(body.text, voice=voice)
        if len(wav_bytes) < 500:
            raise RuntimeError("TTS output too small")
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as exc:
        logger.exception("TTS failed: %s", exc)
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc


@router.post(
    "/consult",
    summary="Full voice pipeline: ASR → RAG → LLM → TTS",
    response_class=Response,
)
async def consult(
    audio: Annotated[UploadFile, File(...)],
    conversation_id: Optional[str] = Form(None),
    current_user: Optional[dict] = Depends(get_optional_user),
) -> Response:
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        voice = current_user.get("tts_voice", "Ezinne") if current_user else "Ezinne"
        # Use only the current conversation's history so the LLM has coherent context
        history = db_get_conversation(conversation_id) if conversation_id else []
        wav_bytes, meta = run_voice_consult(audio_bytes, audio.content_type, voice=voice, history=history)

        if current_user:
            db_save_consultation(
                current_user["id"],
                meta.get("transcript", ""),
                meta.get("guidance", ""),
                meta.get("escalate", False),
                conversation_id=conversation_id,
            )

        import urllib.parse
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-VoiceMed-Transcript": urllib.parse.quote(meta.get("transcript", "")[:200]),
                "X-VoiceMed-Escalate": str(meta.get("escalate", False)).lower(),
                "X-VoiceMed-Guidance": urllib.parse.quote(meta.get("guidance", "")),
                "X-VoiceMed-ConversationId": conversation_id or "",
                "Content-Length": str(len(wav_bytes)),
                "Cache-Control": "no-store",
            },
        )

    except ValueError as exc:
        logger.warning("Consult bad request: %s", exc)
        voice = current_user.get("tts_voice", "Ezinne") if current_user else "Ezinne"
        return Response(
            content=synthesize_speech(
                "We no fit hear you well. Please hold the microphone and talk again.",
                voice=voice,
            ),
            media_type="audio/wav",
        )
    except RuntimeError as exc:
        logger.warning("Consult speech error: %s", exc)
        msg = str(exc).lower()
        msg_text = (
            "We no fit hear you well. Hold the mic button, talk for at least two seconds, then tap again."
            if ("short" in msg or "no speech" in msg)
            else "Speech recognition no work this time. Please try again, or ask the CHEW for help."
        )
        voice = current_user.get("tts_voice", "Ezinne") if current_user else "Ezinne"
        return Response(content=synthesize_speech(msg_text, voice=voice), media_type="audio/wav")
    except Exception as exc:
        logger.exception("Consult pipeline failed: %s", exc)
        voice = current_user.get("tts_voice", "Ezinne") if current_user else "Ezinne"
        return Response(content=synthesize_speech(_ERROR_VOICE, voice=voice), media_type="audio/wav")

