"""Application configuration for VoiceMedAI local deployment."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = Path(os.getenv("VOICEMED_DB_PATH", str(DATA_DIR / "voicemed.db")))
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set in .env — refusing to start without a signing key")

IDIOMS_PATH = DATA_DIR / "idioms.json"
KNOWLEDGE_PATH = DATA_DIR / "phc_knowledge.json"
KNOWLEDGE_CSV_PATH = ROOT_DIR / "train.csv"
ESCALATION_PATH = DATA_DIR / "escalation_keywords.json"
FAISS_INDEX_PATH = INDEX_DIR / "phc.faiss"
FAISS_META_PATH = INDEX_DIR / "phc_meta.json"

AUDIO_CACHE_DIR = Path(os.getenv("AUDIO_CACHE_DIR", str(ROOT_DIR / "backend" / "cache" / "audio")))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


def _extract_whisper_size(model_id: str) -> str:
    if "whisper-" in model_id:
        return model_id.rsplit("whisper-", 1)[-1].strip()
    return "tiny"


# ASR: auto | whisper | demo
ASR_MODE = os.getenv("VOICEMED_ASR_MODE", "auto")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "openai/whisper-small")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE") or _extract_whisper_size(HF_MODEL_ID)


# RAG (FAISS + sentence-transformers per design spec)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RAG_USE_FAISS = os.getenv("RAG_USE_FAISS", "true").lower() in ("1", "true", "yes")

# LLM (llama.cpp GGUF)
LOCAL_LLM_MODEL_PATH = Path(
    os.getenv("LOCAL_LLM_MODEL_PATH", str(MODELS_DIR / "llm.gguf"))
)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
LLM_ENABLED = os.getenv("LLM_ENABLED", "auto").lower()  # auto | on | off
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# TTS: sapi | pyttsx3 | coqui | auto
TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")

# Honorifics (FR-08)
HONORIFICS = ("Oga", "Madam", "Bros", "Sister")

