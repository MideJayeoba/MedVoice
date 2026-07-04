"""Triage prediction — SVM models for department, disease category, priority.

The bundle at models/triage_models.joblib holds three sklearn pipelines
(TfidfVectorizer + LinearSVC) trained on patient complaint text.

The pickled vectorizers reference `clean_medical_text` from `__main__`
(they were trained in a script), so we must inject that function into
`__main__` before joblib.load can unpickle them.

Loading takes ~100 s, so it happens in a background thread at startup.
Until it finishes, predict_triage() returns None and the app works
without triage — never block a consultation on model load.
"""

import logging
import re
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# Prefer the compact retrained bundle (81 MB, loads in seconds);
# fall back to the original 2.5 GB bundle if it's not there.
_SMALL = _MODELS_DIR / "triage_models_small.joblib"
_FULL = _MODELS_DIR / "triage_models.joblib"
MODEL_PATH = _SMALL if _SMALL.exists() else _FULL

_models: dict | None = None
_load_lock = threading.Lock()
_load_started = False
_load_error: str | None = None


def clean_medical_text(text):
    """Must match the preprocessor used at training time exactly."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _load():
    global _models, _load_error
    try:
        # The pickle references clean_medical_text on __main__
        sys.modules["__main__"].clean_medical_text = clean_medical_text
        import joblib

        logger.info("Loading triage models from %s ...", MODEL_PATH)
        _models = joblib.load(MODEL_PATH)
        logger.info("Triage models ready: %s", list(_models.keys()))
    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Triage model load failed: %s", exc)


def preload_triage() -> None:
    """Kick off background load — call once at startup."""
    global _load_started
    with _load_lock:
        if _load_started:
            return
        _load_started = True
    if not MODEL_PATH.exists():
        logger.warning("Triage model file missing: %s — triage disabled", MODEL_PATH)
        return
    threading.Thread(target=_load, daemon=True, name="triage-loader").start()


def get_triage_status() -> dict:
    return {
        "ready": _models is not None,
        "loading": _load_started and _models is None and _load_error is None,
        "error": _load_error,
        "model_path": str(MODEL_PATH),
    }


def _predict_one(pipe, text: str) -> tuple[str, float, float, float]:
    """Returns (label, confidence, margin, top_score).

    Confidence derives from the top-2 decision-score margin: LinearSVC has
    no probabilities, but a wide gap between the best and second-best class
    means an unambiguous prediction. margin/(1+margin) squashes to (0, 1).
    Empirical calibration on this model: clear complaints score margins
    1.0-2.6, ambiguous ones 0.2-0.9, off-topic text under 0.1.
    """
    label = pipe.predict([text])[0]
    scores = np.sort(np.asarray(pipe.decision_function([text])[0], dtype=np.float64).ravel())[::-1]
    if scores.size == 1:  # binary margin
        margin, top = abs(scores[0]), scores[0]
    else:
        margin, top = scores[0] - scores[1], scores[0]
    return str(label), float(margin / (1.0 + margin)), float(margin), float(top)


def _confidence_band(margin: float) -> str:
    if margin >= 0.8:
        return "high"
    if margin >= 0.25:
        return "medium"
    return "low"


# Red-flag phrases that must always escalate to Emergency regardless of the
# classifier — its Emergency recall is ~0.53, too low to trust alone.
_EMERGENCY_PATTERNS = re.compile(
    r"\b(convuls|seizure|shaking|unconscious|not breathing|cant breathe|can t breathe|"
    r"chest pain.{0,30}(arm|numb)|(arm|face).{0,30}numb.{0,30}chest|"
    r"suicid|kill myself|hurt myself|"
    r"drank (kerosene|poison|bleach|acid)|swallowed|overdose|"
    r"bleeding (a lot|heavily|seriously|too much)|blood every ?where|"
    r"snake ?bite|collaps|faint|stroke|"
    r"baby.{0,40}(not|no).{0,10}mov)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Relevance gate — is the text actually about health?
# ---------------------------------------------------------------------------
# A LinearSVC always returns *some* class, so a greeting like "hey" still
# gets mapped to a department. We refuse to auto-predict until the text
# carries real medical signal. Words are split into two strengths:
#   STRONG — symptoms, conditions, mental-health, medications. One is enough
#            to auto-predict ("my chest hurts", "I feel dizzy", "fever").
#   WEAK   — bare body parts. On their own they are ambiguous ("my back was
#            against the wall"), so two are needed, or one alongside a strong.
# Greetings / small-talk contain neither and never auto-trigger.
MIN_MEDICAL_HITS = 2  # weak-only words needed to auto-predict

_SYMPTOM_TERMS = frozenset({
    # symptoms
    "pain", "ache", "aches", "aching", "hurt", "hurts", "hurting", "sore",
    "headache", "migraine", "fever", "feverish", "temperature", "chills",
    "cough", "coughing", "cold", "catarrh", "sneeze", "sneezing", "congestion",
    "vomit", "vomiting", "nausea", "nauseous", "diarrhoea", "diarrhea", "purge",
    "purging", "stool", "constipation", "constipated", "bloating", "bloated",
    "cramp", "cramps", "cramping", "spasm", "dizzy", "dizziness", "faint",
    "fainting", "weak", "weakness", "fatigue", "tired", "exhausted",
    "rash", "rashes", "itching", "itch", "itchy", "swelling", "swollen",
    "swell", "bleeding", "blood", "bloody", "discharge", "burning", "burn",
    "numb", "numbness", "tingling", "breathless", "wheezing", "wheeze",
    "palpitation", "palpitations", "seizure",
    "seizures", "convulsion", "convulsions", "convulsing", "ulcer", "boil",
    "blister", "blisters", "jaundice", "stiff", "stiffness", "wound", "cut",
    "fracture", "fractured", "sprain", "sprained", "bruise", "bruised",
    "sweating", "sweats", "infected", "infection", "lump", "lumps",
    "unwell", "sick", "ill", "illness", "disease", "symptom", "symptoms",
    "vomited", "bleed", "bleeds", "shivering", "collapse", "collapsed",
    "hurting", "pains", "painful", "paining", "runny", "watery", "blocked",
    "discharging", "twisted", "twist", "injured", "injury", "injuries",
    "dislocated", "pulled", "strain", "scratching", "scratch", "peeling",
    "bump", "bumps", "pimple", "pimples", "acne", "spots", "sores", "wounds",
    "cuts", "bite", "bites", "burns", "scald", "scalded", "ringworm",
    "smelly", "stooling", "diarhea", "loose", "gas", "indigestion",
    # conditions / diseases
    "malaria", "typhoid", "diabetes", "diabetic", "hypertension", "asthma",
    "asthmatic", "pneumonia", "cancer", "tumor", "tumour", "hepatitis", "hiv",
    "std", "sti", "arthritis", "allergy", "allergic", "allergies", "stroke",
    "epilepsy", "pregnancy", "pregnant", "menstruation", "menstrual", "period",
    "periods", "miscarriage", "labour", "contraception", "menopause",
    # mental health
    "sad", "depressed", "depression", "anxious", "anxiety", "worried",
    "stressed", "stress", "hopeless", "suicidal", "suicide", "panic",
    "insomnia", "sleepless", "mood", "crying", "lonely",
    "restless", "overthinking",
    # medications / care
    "paracetamol", "panadol", "amoxicillin", "ampiclox", "flagyl", "septrin",
    "metronidazole", "antibiotic", "antibiotics", "medicine",
    "medication", "tablet", "tablets", "injection", "ointment", "cream",
    "syrup", "dose", "dosage", "prescription",
    # pidgin health words
    "pikin",
})

_BODY_TERMS = frozenset({
    "head", "chest", "stomach", "belly", "belle", "abdomen", "abdominal",
    "waist", "back", "leg", "legs", "arm", "arms", "hand", "hands", "foot",
    "feet", "knee", "knees", "ankle", "shoulder", "elbow", "wrist", "neck",
    "throat", "ear", "ears", "eye", "eyes", "nose", "mouth", "tooth", "teeth",
    "gum", "gums", "skin", "scalp", "hair", "nail", "nails", "tongue", "lip",
    "lips", "breast", "breasts", "testicle", "testicles", "scrotum", "penis",
    "vagina", "vaginal", "urine", "urinate", "urinating", "kidney", "liver",
    "lung", "lungs", "heart", "bone", "bones", "joint", "joints",
    "muscle", "muscles", "tummy", "sinus", "breath", "breathing", "sleep",
    "appetite", "weight",
})

_MEDICAL_TERMS = _SYMPTOM_TERMS | _BODY_TERMS


def medical_signal_count(text: str) -> int:
    """Count words that appear in the medical lexicon (relevance signal)."""
    if not text:
        return 0
    words = clean_medical_text(text).split()
    return sum(1 for w in words if w in _MEDICAL_TERMS)


def has_medical_signal(text: str) -> bool:
    """True when the text is health-related enough to auto-predict.

    One strong symptom word is enough; bare body parts need MIN_MEDICAL_HITS.
    """
    if not text:
        return False
    words = clean_medical_text(text).split()
    strong = sum(1 for w in words if w in _SYMPTOM_TERMS)
    weak = sum(1 for w in words if w in _BODY_TERMS)
    return strong >= 1 or (strong + weak) >= MIN_MEDICAL_HITS


def has_emergency_signal(text: str) -> bool:
    """True if the text contains a red-flag phrase that must escalate now."""
    if not text:
        return False
    return bool(_EMERGENCY_PATTERNS.search(clean_medical_text(text)))


def predict_triage(text: str) -> dict | None:
    """Predict department/category/priority for patient complaint text.

    Returns None when the models aren't loaded yet, text is empty, or the
    complaint looks off-topic (rejection threshold) — callers must treat
    triage as optional.
    """
    if _models is None or not text or not text.strip():
        return None

    cleaned = clean_medical_text(text)
    try:
        category, cat_conf, cat_margin, cat_top = _predict_one(_models["category"], cleaned)
        department, dept_conf, dept_margin, _ = _predict_one(_models["department"], cleaned)
        priority, prio_conf, prio_margin, _ = _predict_one(_models["priority"], cleaned)
    except Exception as exc:
        logger.exception("Triage prediction failed: %s", exc)
        return None

    # Rejection: non-medical text scores far from every class with no margin
    if cat_top < -0.5 and cat_margin < 0.15:
        logger.info("Triage rejected (off-topic): %s", text[:60])
        return None

    # Safety net: red-flag keywords force Emergency (escalate-only)
    if priority != "Emergency" and _EMERGENCY_PATTERNS.search(cleaned):
        logger.warning("Triage escalated to Emergency by red-flag keywords: %s", text[:80])
        priority, prio_conf, prio_margin = "Emergency", 0.9, 2.0

    overall = min(cat_conf, dept_conf, prio_conf)
    overall_margin = min(cat_margin, dept_margin, prio_margin)
    result = {
        "category": category,
        "category_confidence": round(cat_conf, 3),
        "department": department,
        "department_confidence": round(dept_conf, 3),
        "priority": priority,
        "priority_confidence": round(prio_conf, 3),
        "confidence": round(overall, 3),
        "confidence_band": _confidence_band(overall_margin),
    }
    logger.info(
        "Triage: %s / %s / %s (conf %.2f, %s)",
        category, department, priority, overall, result["confidence_band"],
    )
    return result
