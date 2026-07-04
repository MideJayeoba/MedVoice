"""LLM reasoning — Groq or Gemini cloud API."""

import logging
import os
import httpx
from backend.config import LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# ── API keys (explicit names only — avoid generic "API_KEY") ──────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = (
    "You are Priscilla, a warm, trusted community health assistant for Primary "
    "Health Centres in Nigeria. Speak in simple, friendly Nigerian English that is "
    "easy to read aloud, and keep every reply under 90 words.\n\n"
    "Your approach is FIRST AID FIRST:\n"
    "1. Begin with a short reassuring word, then give one or two simple, safe "
    "first-aid or home-care steps the patient can do right now for relief.\n"
    "2. Then ask ONE short follow-up question to understand the problem better.\n"
    "3. Do NOT give a definitive diagnosis, and do NOT rush the patient to a "
    "hospital or a specialist for mild or moderate complaints. Never name a medical "
    "department or specialist unless it is truly serious.\n"
    "4. Tell them to visit the health centre only if the problem does not improve in "
    "a day or two, gets worse, or shows danger signs.\n\n"
    "SAFETY OVERRIDE (this beats the first-aid steps above): if the message shows an "
    "emergency — severe bleeding, trouble breathing, chest pain, numb arm/face, "
    "convulsions, poisoning, stroke signs, thoughts of self-harm, or similar — your "
    "FIRST sentence must firmly tell them to go to the nearest hospital or call for "
    "emergency help right now. Do not ask a casual follow-up question or delay. If a "
    "question is not about health, gently say you can only help with health "
    "questions. Never recommend dangerous self-medication."
)


def _triage_hint(triage: dict | None) -> str:
    """Render the ML triage prediction as PRIVATE background for the LLM.

    The prediction must never be announced to the patient (no "go to a
    neurologist"). It only nudges how urgent and how confident the advice
    should feel. First-aid stays the default for non-urgent cases.
    """
    if not triage:
        return ""
    priority = triage.get("priority", "Moderate")
    band = triage.get("confidence_band", "low")
    urgent = priority in ("Emergency", "High")

    if urgent:
        action = (
            "This is a possible EMERGENCY. Your FIRST sentence must tell the patient "
            "to go to the nearest hospital or call for emergency help right now — do "
            "not delay and do NOT ask a casual follow-up question. You may then add "
            "one or two safe things to do while they get help. This overrides the "
            "first-aid-first style."
        )
    else:
        action = (
            "This is NOT urgent. Lead with simple first-aid / home-care steps for "
            "relief, then ask ONE short follow-up question. Do NOT conclude, do NOT "
            "name any department or specialist, and do NOT send them to a doctor now "
            "— only say to visit the health centre if it does not improve in a day or "
            "two or gets worse."
        )

    confidence = {
        "high": "You may be reasonably clear about the likely problem, but stay preliminary.",
        "medium": "Stay tentative in your wording.",
        "low": "Be clearly unsure; rely on your follow-up question, not the guess.",
    }[band]

    return (
        "\n\n[INTERNAL TRIAGE HINT — never read these labels aloud or mention any "
        f"department/specialist name.] A support model suggests the complaint may "
        f"involve the '{triage['category']}' area, with urgency '{priority}'. Use "
        f"this only to shape safe advice. {action} {confidence}"
    )


def _build_messages(query: str, history: list[dict] | None, triage: dict | None = None) -> list[dict]:
    """Build chat message list with optional prior consultation context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + _triage_hint(triage)}]
    for item in (history or [])[-5:]:  # include up to 5 prior turns
        if item.get("transcript"):
            messages.append({"role": "user", "content": item["transcript"]})
        if item.get("guidance"):
            messages.append({"role": "assistant", "content": item["guidance"]})
    messages.append({"role": "user", "content": query})
    return messages


def _call_groq(query: str, history: list[dict] | None = None, triage: dict | None = None) -> str | None:
    """Call Groq API (Llama 3.1 8B). Returns text or None on failure."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": _build_messages(query, history, triage),
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    try:
        logger.info("Calling Groq API for: %s", query[:80])
        r = httpx.post(url, headers=headers, json=payload, timeout=15.0)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text:
                logger.info("Groq inference OK (%d chars)", len(text))
                return text
        else:
            logger.error("Groq API %s: %s", r.status_code, r.text[:300])
    except Exception as exc:
        logger.exception("Groq call failed: %s", exc)
    return None


def _call_gemini(query: str, history: list[dict] | None = None, triage: dict | None = None) -> str | None:
    """Call Gemini 2.5 Flash Lite. Returns text or None on failure."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    contents = []
    for item in (history or [])[-5:]:
        if item.get("transcript"):
            contents.append({"role": "user", "parts": [{"text": item["transcript"]}]})
        if item.get("guidance"):
            contents.append({"role": "model", "parts": [{"text": item["guidance"]}]})
    contents.append({"role": "user", "parts": [{"text": query}]})
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + _triage_hint(triage)}]},
        "generationConfig": {
            "temperature": LLM_TEMPERATURE,
            "maxOutputTokens": LLM_MAX_TOKENS,
        },
    }
    try:
        logger.info("Calling Gemini API for: %s", query[:80])
        r = httpx.post(url, json=payload, timeout=15.0)
        if r.status_code == 200:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text:
                logger.info("Gemini inference OK (%d chars)", len(text))
                return text
        else:
            logger.error("Gemini API %s: %s", r.status_code, r.text[:300])
    except Exception as exc:
        logger.exception("Gemini call failed: %s", exc)
    return None


FALLBACK = (
    "Thank you for sharing your concern. "
    "Visit the nearest healthcare center so they can check you properly if urgent. "
    "We will get back to you as soon as possible."
)


def generate_guidance(query: str, history: list[dict] | None = None, triage: dict | None = None) -> str:
    """Send transcript to cloud LLM and return the response text."""
    if GROQ_API_KEY:
        result = _call_groq(query, history, triage)
        if result:
            return result

    if GEMINI_API_KEY:
        result = _call_gemini(query, history, triage)
        if result:
            return result

    logger.warning("No LLM API key configured or all calls failed — using fallback")
    return FALLBACK


def get_reasoning_status() -> dict:
    """Report which LLM provider is active."""
    if GROQ_API_KEY:
        mode, model = "groq-api", "llama-3.1-8b-instant"
    elif GEMINI_API_KEY:
        mode, model = "gemini-api", "gemini-2.5-flash-lite"
    else:
        mode, model = "rules", "fallback-rules"
    ready = bool(GROQ_API_KEY or GEMINI_API_KEY)
    return {
        "llm": {
            "enabled": True,
            "model_path": model,
            "loaded": ready,
            "exists": True,
            "ready": ready,
        },
        "mode": mode,
    }

