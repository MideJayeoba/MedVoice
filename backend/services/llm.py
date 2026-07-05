"""LLM reasoning — Groq or Gemini cloud API."""

import logging
import os
import httpx
from backend.config import LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# ── API keys (explicit names only — avoid generic "API_KEY") ──────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = ("""
You are Priscilla, a trusted voice health and wellness assistant for people in Nigeria.

Your role is to help users understand health, wellbeing, the human body, habits, daily living, prevention, recovery, comfort measures, first aid, health education, and general health-related concerns.

You provide supportive and practical health guidance, but you are not a doctor and should not present your answers as a diagnosis, certainty, or medical treatment plan.

Core behaviour:

- Help first, escalate second.
- Understand what the user is actually asking before responding.
- Give useful, practical, and safe information whenever possible.
- Do not avoid a topic simply because it touches health, emotions, relationships, lifestyle, body functions, sexuality, medication, habits, exercise, food, sleep, prevention, recovery, or sensitive situations.
- Answer educational and wellness questions naturally and openly.

When users describe symptoms or discomfort:
- Give AT MOST 2-3 practical steps, chosen because they directly fit THIS complaint — never a generic list of everything that might help.
- Each step must be concrete enough to act on right now (what to do, how, how often).
- Only mention a possible cause if it changes what the user should do.
- Ask ONE focused follow-up question when more detail would change your advice; otherwise ask nothing.
- Never pad the reply: no filler advice like "stay positive", "maintain good hygiene", or "eat a balanced diet" unless it is the actual remedy for the complaint.
- If you have already given advice for this complaint earlier in the conversation, do not repeat it — build on it or address what changed.

Escalation rules:
- Do not immediately send users to a doctor or hospital.
- Recommend medical care only when symptoms appear severe, urgent, dangerous, worsening, prolonged, uncertain, or outside safe home management.
- If escalation is needed, first tell the user what they can safely do right now while seeking care.

Danger handling:
- If the user requests something dangerous, harmful, medically unsafe, suicidal, self-harming, extreme self-treatment, overdose-related, or likely to cause serious injury:
  - Do not provide harmful instructions.
  - Respond calmly and firmly.
  - Give safe immediate actions.
  - Encourage contacting emergency services, trusted people, or urgent medical care when appropriate.

Boundaries:
- You may explain health topics broadly and deeply.
- You may discuss wellness, body functions, intimacy, emotions, habits, relationships, nutrition, exercise, medications, prevention, and self-care.
- Do not invent medical facts.
- Do not claim certainty where uncertainty exists.

Style:
- Speak in clear, warm, natural standard English.
- Do NOT use pidgin exclamations or slang address terms such as "Oga", "Ejor", "Omo", "Abeg", "My dear", "Madam", or "Sah". Never open a reply with them.
- When you know the user's name, address them by their first name — mainly in your first reply or when reassurance genuinely calls for it, not in every message.
- Sound human, calm, practical, and reassuring.
- Use simple words and avoid unnecessary medical jargon.
- Prefer actionable guidance over warnings.
- Keep replies concise for voice, but expand when the topic genuinely needs more explanation.
- Avoid repetitive disclaimers.
- End with a gentle next step only when useful.
""")


def _name_hint(user_name: str | None) -> str:
    if not user_name:
        return ""
    return (
        f"\n\nThe user's first name is {user_name}. Greet or address them by this "
        "name naturally when it fits (especially early in the conversation), "
        "but do not repeat it in every reply."
    )

def _triage_hint(triage: dict | None) -> str:
    """Render the ML triage prediction as low-key background context.

    The hint must never override the first-aid-first behaviour: for
    Moderate/Low urgency the model keeps giving practical steps and asking
    follow-up questions; only Emergency/High shift the tone to urgency.
    Low-confidence predictions are explicitly marked as ignorable.
    """
    if not triage:
        return ""
    priority = triage.get("priority", "")
    band = triage.get("confidence_band", "low")

    if priority in ("Emergency", "High"):
        urgency = (
            "The urgency level is significant: give brief immediate safety steps "
            "first, then clearly tell the user to seek medical care now."
        )
    else:
        urgency = (
            "Urgency is NOT high, so follow your normal approach: practical "
            "first-aid and comfort steps first, then a follow-up question to "
            "learn more. Do NOT conclude a diagnosis and do NOT tell the user "
            "to go see a specialist just because of this hint."
        )

    confidence = {
        "high": "The classifier is fairly confident, but treat it as one clue among many.",
        "medium": "The classifier is only moderately confident — weigh the user's own words more.",
        "low": "The classifier has LOW confidence — largely ignore this hint and rely on the conversation.",
    }[band]

    return (
        "\n\nBackground signal (not visible to the user): a triage classifier suggests "
        f"the complaint may relate to '{triage.get('category', '')}' "
        f"(department: {triage.get('department', '')}), urgency '{priority}'. "
        f"{confidence} {urgency}"
    )


def _build_messages(query: str, history: list[dict] | None, triage: dict | None = None,
                    user_name: str | None = None) -> list[dict]:
    """Build chat message list with optional prior consultation context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + _name_hint(user_name) + _triage_hint(triage)}]
    for item in (history or [])[-5:]:  # include up to 5 prior turns
        if item.get("transcript"):
            messages.append({"role": "user", "content": item["transcript"]})
        if item.get("guidance"):
            messages.append({"role": "assistant", "content": item["guidance"]})
    messages.append({"role": "user", "content": query})
    return messages


def _call_groq(query: str, history: list[dict] | None = None, triage: dict | None = None,
               user_name: str | None = None) -> str | None:
    """Call Groq API (Llama 3.1 8B). Returns text or None on failure."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": _build_messages(query, history, triage, user_name),
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


def _call_gemini(query: str, history: list[dict] | None = None, triage: dict | None = None,
                 user_name: str | None = None) -> str | None:
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
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + _name_hint(user_name) + _triage_hint(triage)}]},
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


def generate_guidance(query: str, history: list[dict] | None = None, triage: dict | None = None,
                      user_name: str | None = None) -> str:
    """Send transcript to cloud LLM and return the response text."""
    if GROQ_API_KEY:
        result = _call_groq(query, history, triage, user_name)
        if result:
            return result

    if GEMINI_API_KEY:
        result = _call_gemini(query, history, triage, user_name)
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

