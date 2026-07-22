#!/usr/bin/env python
"""Screenshot-ready demonstrations of each prediction component.

Prints clean, labelled terminal output suitable for a project report.
Application logs are silenced so only the demonstration shows.

Usage:
    python scripts/demo_predictions.py <component>

Components:
    category    Disease category classifier (68 classes, LinearSVC)
    priority    Urgency classifier (Emergency/High/Moderate/Low)
    department  Category -> specialist mapping (deterministic lookup)
    triage      Full triage output for sample complaints
    gate        Relevance gate + emergency red-flag safety net
    llm         LLM guidance + symptom enrichment (Groq Llama 3.1 8B)
    asr         Speech-to-text (Groq Whisper large-v3-turbo)
    tts         Text-to-speech (Edge-TTS, Nigerian voices)
    pipeline    Full end-to-end consultation
    all         Every component in sequence
"""

import logging
import sys
import time
from pathlib import Path

# Silence application logs so screenshots contain only the demo output
logging.disable(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W = 78


def header(title: str, subtitle: str = "") -> None:
    print()
    print("=" * W)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * W)


def rule() -> None:
    print("-" * W)


def load_models():
    from backend.services.triage import preload_triage, get_triage_status, MODEL_PATH
    t0 = time.time()
    preload_triage()
    while not get_triage_status()["ready"]:
        time.sleep(0.2)
    return MODEL_PATH.name, time.time() - t0


# ---------------------------------------------------------------------------
# 1. Disease category classifier
# ---------------------------------------------------------------------------
def demo_category():
    from backend.services.triage import _models, _predict_one, clean_medical_text
    name, secs = load_models()
    from backend.services.triage import _models as M

    header("COMPONENT 1: DISEASE CATEGORY CLASSIFIER",
           "Model: TF-IDF (1,2-grams) + LinearSVC  |  68 classes")
    clf = M["category"].named_steps["clf"]
    vec = M["category"].named_steps["tfidf"]
    print(f"  Bundle          : {name}")
    print(f"  Load time       : {secs:.2f} s")
    print(f"  Output classes  : {len(clf.classes_)}")
    print(f"  TF-IDF features : {len(vec.vocabulary_):,}")
    rule()
    print(f"  {'PATIENT COMPLAINT':<44}{'PREDICTED CATEGORY':<26}{'CONF'}")
    rule()
    samples = [
        "red itchy rash spreading on my arm",
        "my tooth has been aching for two days",
        "I twisted my ankle playing football",
        "I keep forgetting things and names",
        "I feel sad and I cannot sleep",
        "chest feels tight, left arm is numb",
        "coughing with catarrh for a week",
        "burning feeling when I urinate",
    ]
    for s in samples:
        label, conf, _margin, _top = _predict_one(M["category"], clean_medical_text(s))
        print(f"  {s:<44}{label[:24]:<26}{conf:.2f}")
    rule()


# ---------------------------------------------------------------------------
# 2. Priority classifier
# ---------------------------------------------------------------------------
def demo_priority():
    from backend.services.triage import _predict_one, clean_medical_text
    name, secs = load_models()
    from backend.services.triage import _models as M

    header("COMPONENT 2: URGENCY / PRIORITY CLASSIFIER",
           "Model: TF-IDF (1,2-grams) + LinearSVC  |  4 levels")
    clf = M["priority"].named_steps["clf"]
    vec = M["priority"].named_steps["tfidf"]
    print(f"  Output classes  : {list(clf.classes_)}")
    print(f"  TF-IDF features : {len(vec.vocabulary_):,}")
    rule()
    print(f"  {'PATIENT COMPLAINT':<48}{'PRIORITY':<14}{'CONF'}")
    rule()
    samples = [
        "my child is convulsing and not breathing",
        "I am bleeding heavily and feeling faint",
        "chest is tight and my arm feels numb",
        "fever and headache for three days",
        "my tooth has been aching for two days",
        "advice on healthy eating and weight",
    ]
    for s in samples:
        label, conf, _m, _t = _predict_one(M["priority"], clean_medical_text(s))
        print(f"  {s:<48}{label:<14}{conf:.2f}")
    rule()


# ---------------------------------------------------------------------------
# 3. Department lookup
# ---------------------------------------------------------------------------
def demo_department():
    from backend.services.triage import department_for_category, _DEPT_MAP
    header("COMPONENT 3: DEPARTMENT / SPECIALIST ASSIGNMENT",
           "Deterministic lookup: category -> specialist (no ML inference)")
    print(f"  Mapping source  : data/category_to_department.json")
    print(f"  Categories      : {len(_DEPT_MAP)}")
    print(f"  Distinct depts  : {len(set(_DEPT_MAP.values()))}")
    rule()
    print(f"  {'PREDICTED CATEGORY':<34}{'ASSIGNED SPECIALIST'}")
    rule()
    for cat in ["Toothache", "Skin issue", "Ankle pain", "Memory disturbance",
                "Emotional pain", "Heart hurts", "Eye Infection", "Ear ache",
                "Stomach ache", "Cough"]:
        print(f"  {cat:<34}{department_for_category(cat)}")
    rule()


# ---------------------------------------------------------------------------
# 4. Full triage
# ---------------------------------------------------------------------------
def demo_triage():
    from backend.services.triage import predict_triage
    load_models()
    header("COMPONENT 4: COMPLETE TRIAGE OUTPUT",
           "Category + Priority + Department + Confidence band")
    samples = [
        "my tooth has been aching for two days and my gum is swollen",
        "I twisted my ankle at football and it is swollen",
        "my child has high fever and is shaking",
        "I feel sad and hopeless and I cannot sleep at night",
    ]
    for s in samples:
        r = predict_triage(s)
        rule()
        print(f"  INPUT      : {s}")
        if not r:
            print("  OUTPUT     : (no prediction - not health related)")
            continue
        print(f"  Category   : {r['category']}")
        print(f"  Department : {r['department']}")
        print(f"  Priority   : {r['priority']}")
        print(f"  Confidence : {r['confidence']:.3f}  ({r['confidence_band']})")
    rule()


# ---------------------------------------------------------------------------
# 5. Safety gates
# ---------------------------------------------------------------------------
def demo_gate():
    from backend.services.triage import (has_medical_signal, has_emergency_signal,
                                         predict_triage)
    load_models()
    header("COMPONENT 5: RELEVANCE GATE & EMERGENCY SAFETY NET",
           "Prevents predictions on non-health text; forces Emergency on red flags")
    print(f"  {'INPUT TEXT':<44}{'MEDICAL?':<11}{'RED FLAG?':<11}{'PREDICTS?'}")
    rule()
    samples = [
        "hey", "hello how are you", "what is the weather today",
        "my chest hurts when I breathe", "I have fever and headache",
        "my child is convulsing", "I took too many tablets",
    ]
    for s in samples:
        med = has_medical_signal(s)
        red = has_emergency_signal(s)
        r = predict_triage(s) if (med or red) else None
        out = f"{r['priority']}" if r else "no"
        print(f"  {s:<44}{str(med):<11}{str(red):<11}{out}")
    rule()
    print("  Non-health text yields NO prediction, so it cannot influence guidance.")
    print("  Red-flag phrases force Emergency regardless of classifier output.")
    rule()


# ---------------------------------------------------------------------------
# 6. LLM
# ---------------------------------------------------------------------------
def demo_llm():
    from backend.services.llm import generate_guidance
    header("COMPONENT 6: LLM REASONING & SYMPTOM ENRICHMENT",
           "Groq llama-3.1-8b-instant  |  JSON output: guidance + enriched_symptoms")
    samples = [
        ("my tooth has been aching for two days", {"name": "Ada", "age": 28, "gender": "Female"}),
        ("who won the football match last night?", {"name": "Ada"}),
    ]
    for q, prof in samples:
        rule()
        t0 = time.time()
        guidance, enriched = generate_guidance(q, user_name=prof)
        dt = time.time() - t0
        print(f"  USER INPUT       : {q}")
        print(f"  LATENCY          : {dt:.2f} s")
        print(f"  ENRICHED SYMPTOMS: {enriched[:150] or '(none)'}")
        print(f"  SPOKEN GUIDANCE  : {guidance[:300]}")
    rule()


# ---------------------------------------------------------------------------
# 7. ASR
# ---------------------------------------------------------------------------
def demo_asr():
    from backend.services.asr import get_asr_status, transcribe_audio
    from backend.services.tts import synthesize_speech
    header("COMPONENT 7: AUTOMATIC SPEECH RECOGNITION (ASR)",
           "Groq Whisper large-v3-turbo, local Whisper fallback")
    st = get_asr_status()
    for k, v in st.items():
        print(f"  {k:<16}: {v}")
    rule()
    phrase = "I have been having severe headache and fever since yesterday"
    print(f"  Generating test speech: \"{phrase}\"")
    audio = synthesize_speech(phrase, voice="Ezinne")
    print(f"  Audio generated  : {len(audio):,} bytes (MP3)")
    t0 = time.time()
    text = transcribe_audio(audio, "audio/mpeg")
    dt = time.time() - t0
    rule()
    print(f"  SPOKEN INPUT     : {phrase}")
    print(f"  TRANSCRIBED TEXT : {text}")
    print(f"  LATENCY          : {dt:.2f} s")
    rule()


# ---------------------------------------------------------------------------
# 8. TTS
# ---------------------------------------------------------------------------
def demo_tts():
    from backend.services.tts import synthesize_speech, get_tts_status
    header("COMPONENT 8: TEXT-TO-SPEECH (TTS)",
           "Edge-TTS  |  Nigerian English voices")
    for k, v in get_tts_status().items():
        print(f"  {k:<16}: {v}")
    rule()
    text = "Try rinsing your mouth with warm salt water to ease the pain."
    print(f"  INPUT TEXT : {text}")
    rule()
    print(f"  {'VOICE':<14}{'GENDER':<12}{'BYTES':<14}{'LATENCY'}")
    rule()
    for voice, gender in [("Ezinne", "Female"), ("Abeo", "Male")]:
        t0 = time.time()
        audio = synthesize_speech(text, voice=voice)
        print(f"  {voice:<14}{gender:<12}{len(audio):<14,}{time.time()-t0:.2f} s")
    rule()


# ---------------------------------------------------------------------------
# 9. End-to-end
# ---------------------------------------------------------------------------
def demo_pipeline():
    from backend.services.pipeline import run_consult_text
    load_models()
    header("COMPONENT 9: END-TO-END CONSULTATION PIPELINE",
           "Input -> LLM reasoning -> Triage -> Guidance")
    query = "my tooth has been aching for two days and my gum is swollen"
    profile = {"name": "Ada", "age": 28, "gender": "Female"}
    print(f"  PATIENT INPUT : {query}")
    print(f"  PATIENT DATA  : {profile['age']}-year-old {profile['gender']}")
    rule()
    t0 = time.time()
    guidance, triage = run_consult_text(query, history=[], user_name=profile)
    dt = time.time() - t0
    if triage:
        print(f"  TRIAGE  Category   : {triage['category']}")
        print(f"          Department : {triage['department']}")
        print(f"          Priority   : {triage['priority']}")
        print(f"          Confidence : {triage['confidence']:.3f} ({triage['confidence_band']})")
    else:
        print("  TRIAGE  : (none - input not health related)")
    rule()
    print(f"  GUIDANCE: {guidance}")
    rule()
    print(f"  TOTAL RESPONSE TIME: {dt:.2f} s")
    rule()


DEMOS = {
    "category": demo_category, "priority": demo_priority,
    "department": demo_department, "triage": demo_triage,
    "gate": demo_gate, "llm": demo_llm, "asr": demo_asr,
    "tts": demo_tts, "pipeline": demo_pipeline,
}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        for fn in DEMOS.values():
            fn()
        print()
        return
    if arg not in DEMOS:
        print(f"Unknown component: {arg}")
        print(f"Choose from: {', '.join(DEMOS)}, all")
        sys.exit(1)
    DEMOS[arg]()
    print()


if __name__ == "__main__":
    main()
