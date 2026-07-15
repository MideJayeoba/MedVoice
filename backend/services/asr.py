"""ASR module — Groq Whisper API first, local whisper-small fallback."""

import io
import logging
import os
import wave

import httpx
import numpy as np

from backend.config import ASR_MODE, WHISPER_MODEL_SIZE
from backend.services.audio import ensure_wav_bytes

logger = logging.getLogger(__name__)

# Pooled client — avoids a fresh TLS handshake per transcription
_http = httpx.Client(timeout=30.0)

_whisper_model = None
_mode_resolved: str | None = None

# Set to True once Groq returns a quota/rate-limit error so we stop trying
# for the rest of this server session.
_groq_exhausted: bool = False

SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = 8000  # 0.5 seconds at 16 kHz

# ---------------------------------------------------------------------------
# Nigerian medical English initial_prompt
# ---------------------------------------------------------------------------
# Whisper's initial_prompt primes its vocabulary and accent decoding.
# For Nigerian-accented English, the key phonological patterns are:
#   - "th" → "d" or "t"  (this→dis, three→tree, with→wit)
#   - "v" → "b"           (very→berry, fever→feber)
#   - Vowel mergers: /ɪ/→/iː/ (sit→seat), /ʊ/→/uː/
#   - Final consonant dropping (chest→ches, and→an, of→o)
#   - Syllable stress shift (HYpertension, MALaria)
#   - Clear consonants at syllable onset
#
# The prompt MUST look like the first sentences of a real transcript.
# It seeds Whisper with correct spellings of medical terms it will
# encounter, so it maps the accented pronunciation → correct English word.
_NIGERIAN_MEDICAL_PROMPT = (
    "The patient has fever, malaria, typhoid, hypertension, and diabetes. "
    "She feels pain in her chest, abdomen, waist, and joints. "
    "He has difficulty breathing, dizziness, headache, and body weakness. "
    "The child has convulsions, is not breastfeeding, and has diarrhoea. "
    "The symptoms include vomiting, nausea, jaundice, and skin itching. "
    "She has vaginal discharge, irregular menstruation, and is pregnant. "
    "He took paracetamol, amoxicillin, ampiclox, metronidazole, septrin, and flagyl. "
    "The blood pressure is high and the patient needs urgent referral."
)


def _resolve_mode() -> str:
    global _mode_resolved
    if _mode_resolved:
        return _mode_resolved

    import os

    requested = os.getenv("VOICEMED_ASR_MODE", ASR_MODE).lower()
    if requested in ("whisper", "demo"):
        _mode_resolved = requested
        return _mode_resolved

    # If Groq is configured, we can run cloud whisper without local package
    if os.getenv("GROQ_API_KEY"):
        _mode_resolved = "whisper"
        return _mode_resolved

    try:
        import whisper  # noqa: F401

        _mode_resolved = "whisper"
    except ImportError:
        _mode_resolved = "demo"
        logger.warning(
            "Whisper not installed — ASR uses demo mode. "
            "Run: pip install openai-whisper"
        )
    return _mode_resolved


def get_asr_status() -> dict:
    mode = _resolve_mode()
    groq_key = os.getenv("GROQ_API_KEY")
    groq_active = bool(groq_key) and not _groq_exhausted
    engine = "groq-whisper-large-v3-turbo" if groq_active else f"local-whisper-{WHISPER_MODEL_SIZE}"
    return {
        "mode": mode,
        "engine": engine if mode == "whisper" else "demo",
        "groq_available": groq_active,
        "local_model": f"whisper-{WHISPER_MODEL_SIZE}",
        "ready": True,
    }


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    import whisper

    size = WHISPER_MODEL_SIZE
    logger.info("Loading Whisper '%s' (first run downloads ~%s)…", size,
                {"tiny": "75 MB", "base": "145 MB", "small": "466 MB",
                 "medium": "1.5 GB", "large": "3 GB"}.get(size, "unknown size"))
    _whisper_model = whisper.load_model(size)
    logger.info("Whisper '%s' ready", size)
    return _whisper_model


def _wav_bytes_to_numpy(wav_bytes: bytes) -> np.ndarray:
    """Load 16 kHz mono WAV into float32 numpy array (Whisper input format)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


def transcribe_audio(audio_bytes: bytes, content_type: str | None = None) -> str:
    """Transcribe speech. Tries Groq Whisper API first, falls back to local."""
    wav_bytes = ensure_wav_bytes(audio_bytes, content_type)
    mode = _resolve_mode()

    if mode == "whisper":
        # 1. Try Groq cloud ASR (fast, accurate, free tier ~7200 s/day)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and not _groq_exhausted:
            try:
                text = _transcribe_groq(wav_bytes, groq_key)
                logger.info("[Groq ASR] transcribed %d chars", len(text))
                return _post_process_nigerian(text)
            except _GroqQuotaError as exc:
                logger.warning("Groq ASR quota/rate-limit hit — switching to local Whisper: %s", exc)
                globals()['_groq_exhausted'] = True
            except Exception as exc:
                logger.warning("Groq ASR failed, falling back to local Whisper: %s", exc)

        # 2. Local whisper-small fallback
        try:
            return _transcribe_whisper(wav_bytes)
        except Exception as exc:
            logger.exception("Local Whisper also failed: %s", exc)
            raise RuntimeError("Speech recognition failed") from exc

    return _demo_transcribe(wav_bytes)


# ---------------------------------------------------------------------------
# Groq cloud ASR
# ---------------------------------------------------------------------------

class _GroqQuotaError(Exception):
    """Raised on 429 / quota-exceeded so the caller can set the exhausted flag."""


def _transcribe_groq(wav_bytes: bytes, api_key: str) -> str:
    """Send WAV audio to Groq's Whisper API and return the transcript."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav"),
        "model": (None, "whisper-large-v3-turbo"),
        "language": (None, "en"),
        "prompt": (None, _NIGERIAN_MEDICAL_PROMPT[:200]),  # Groq prompt limit
        "response_format": (None, "text"),
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    r = _http.post(url, headers=headers, files=files)

    if r.status_code == 429 or (r.status_code == 400 and "quota" in r.text.lower()):
        raise _GroqQuotaError(f"HTTP {r.status_code}: {r.text[:200]}")
    if not r.is_success:
        raise RuntimeError(f"Groq ASR HTTP {r.status_code}: {r.text[:200]}")

    text = r.text.strip() if r.headers.get("content-type", "").startswith("text") else r.json().get("text", "").strip()
    if not text:
        raise RuntimeError("Groq returned empty transcript")
    return text


def _transcribe_whisper(wav_bytes: bytes) -> str:
    """
    Transcribe using in-memory numpy audio, optimised for Nigerian accented English.
    Avoids whisper.load_audio() which requires ffmpeg on Windows.
    """
    model = _load_whisper()
    audio = _wav_bytes_to_numpy(wav_bytes)

    if len(audio) < MIN_AUDIO_SAMPLES:
        raise RuntimeError("Recording too short — speak for at least one second")

    peak = float(np.max(np.abs(audio)))
    if peak < 0.01:
        raise RuntimeError("No speech detected — check microphone volume")

    # Normalise volume to ensure consistent input levels regardless of mic sensitivity
    if peak > 0:
        audio = audio / peak * 0.95

    result = model.transcribe(
        audio,
        language="en",
        fp16=False,
        task="transcribe",
        initial_prompt=_NIGERIAN_MEDICAL_PROMPT,
        no_speech_threshold=0.5,
        condition_on_previous_text=True,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.2,
    )
    text = (result.get("text") or "").strip()

    if not text:
        raise RuntimeError("No speech detected — please speak louder and try again")

    text = _post_process_nigerian(text)
    logger.info("ASR transcribed %d chars", len(text))  # content redacted (health data)
    return text


# ---------------------------------------------------------------------------
# Post-processing: correct systematic Whisper mishearings of Nigerian speech
# ---------------------------------------------------------------------------
# Whisper tends to "correct" pidgin into standard English words that don't
# match the medical meaning. These rules fix the most common errors.

import re

_NIGERIAN_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # 1. Base word phonetic corrections (run first so auxiliary rules match them)
    (re.compile(r"\bbelly\b", re.I), "belle"),
    (re.compile(r"\b(?:waste|west)\b", re.I), "waist"),
    (re.compile(r"\b(?:picking|peaking|pekin|peking)\b", re.I), "pikin"),
    (re.compile(r"\bstew\b", re.I), "stool"),
    (re.compile(r"\b(?:stewing|storing)\b", re.I), "stooling"),
    (re.compile(r"\b(?:warm\s+eating|womiting)\b", re.I), "vomiting"),
    (re.compile(r"\b(?:soak|sock)\b", re.I), "suck"),
    (re.compile(r"\b(?:tea\s+for|tea\s+fall|teefor)\b", re.I), "tifo"),
    (re.compile(r"\b(?:confulsion|combulsion)\b", re.I), "convulsion"),
    (re.compile(r"\b(?:press|bless)\s*feed(ing)?\b", re.I), r"breastfeed\1"),
    (re.compile(r"\b(?:bref|bret|breadth|breeze)\s+(no|no\s+fit|dey|they|day)\b", re.I), r"breath \1"),

    # 2. Map Whisper mishearings of the auxiliary verb "dey" (often heard as they, day, the, there, standard copulas like is/am/are)
    # when preceded by common pidgin nouns / subjects or followed by pidgin adjectives / predicates.
    (re.compile(r"\b(body|belle|tummy|tommy|chest|head|diarrhoea|stool|pikin|waist|back|joint|bone|throat|ear|eye|skin|rashes|ulcer|asthma|cough|period|infection|seizure)\s+(?:they|day|the|there|is|am|are|was|were)\b", re.I), r"\1 dey"),
    (re.compile(r"\b(I|we|you|he|she|they|dem)\s+(?:they|day|the|there|am|are)\b", re.I), r"\1 dey"),
    (re.compile(r"\b(no)\s+(?:they|day|the|there)\b", re.I), r"\1 dey"),  # "no dey"
    
    # 3. Medical terms phonetic mishearings (typhoid/tifo, malaria, etc.)
    (re.compile(r"\b(?:tyford|tyfor)\b", re.I), "typhoid"),
    (re.compile(r"\b(?:malara|malearia)\b", re.I), "malaria"),
    (re.compile(r"\b(?:diabetis|diabete)\b", re.I), "diabetes"),
    (re.compile(r"\b(?:hyper-tension|hipertension)\b", re.I), "hypertension"),
    (re.compile(r"\b(?:convolsion|convulse)\b", re.I), "convulsion"),
    (re.compile(r"\b(?:parasitamol|paracetamole|palacitamol)\b", re.I), "paracetamol"),
    (re.compile(r"\b(?:ampiclox|ampyclox|ampiclocks)\b", re.I), "ampiclox"),
    (re.compile(r"\b(?:septrin|septin|septrim)\b", re.I), "septrin"),
    (re.compile(r"\b(?:flagyl|flagel|flaggel)\b", re.I), "flagyl"),
    
    # 4. Aspectual particle "don" (often transcribed as done, down, don't)
    # when followed by typical state verbs or adjectives
    (re.compile(r"\b(?:done|down)\s+(swell|big|yellow|reduce|lose|vomit|collapse|faint|hot)\b", re.I), r"don \1"),
    (re.compile(r"\bI\s+(?:done|down|dont|don't)\s+(vomit|collapse|faint|lose|do|purge)\b", re.I), r"I don \1"),
    (re.compile(r"\b(leg|face|hand|tummy|body|belle|weight|period)\s+(?:done|down)\b", re.I), r"\1 don"),
    
    # 5. Reduplication patterns
    (re.compile(r"\bwell\s+well\b", re.I), "well-well"),
    (re.compile(r"\bsmall\s+small\b", re.I), "small-small"),
    
    # Noise/Punctuation cleanups
    (re.compile(r"^[,\.\s\-\?]+"), ""),
]


def _post_process_nigerian(text: str) -> str:
    """Apply lightweight rule-based corrections for Nigerian ASR output."""
    for pattern, replacement in _NIGERIAN_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _demo_transcribe(wav_bytes: bytes) -> str:
    """Fallback when Whisper is not installed."""
    duration_ms = 2000
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            duration_ms = int((wf.getnframes() / max(wf.getframerate(), 1)) * 1000)
    except Exception:
        duration_ms = max(1000, len(wav_bytes) // 32)

    samples = [
        "body dey hot and belle dey pain me, i think na malaria",
        "chest dey pepper me and breath no dey come well",
        "how many paracetamol i fit take for headache",
        "my pikin no dey suck breast well since yesterday",
        "i dey purge since morning and stool dey rush me",
    ]
    text = samples[(duration_ms // 1000) % len(samples)]
    logger.warning("DEMO ASR (install openai-whisper for real transcription): %d chars", len(text))
    return text
