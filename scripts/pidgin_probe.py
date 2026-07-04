#!/usr/bin/env python
"""Probe: does the triage model understand Nigerian Pidgin / colloquial phrasing?

Each probe pairs a formal-English complaint with a pidgin rendering of the
SAME complaint. If the model gets the formal one right but the pidgin one
wrong (or with far less confidence), the training data needs pidgin coverage.

Usage: python scripts/pidgin_probe.py [model_path]
"""

import re
import sys
from pathlib import Path

import numpy as np


def clean_medical_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# (formal, pidgin, expected_department)
PROBES = [
    ("My child has a high fever and is convulsing",
     "my pikin body dey hot well well and e dey shake convulsion", "Paediatrics"),
    ("I have chest pain and difficulty breathing",
     "my chest dey pain me and breath no dey come well", "Cardiology"),
    ("I have been vomiting and have severe stomach pain",
     "i dey vomit since morning and my belle dey pain me well well", "Gastroenterology"),
    ("I have a skin rash that itches badly",
     "one kind rash full my body and e dey scratch me well well", "Dermatology"),
    ("I am pregnant and bleeding",
     "i get belle and blood dey comot from my body", "Obstetrics & Gynaecology"),
    ("My eyes are red and painful with blurry vision",
     "my eye dey red dey pain me and i no fit see well", "Ophthalmology"),
    ("I have painful urination and blood in my urine",
     "when i wan piss e dey pain me and blood dey inside the piss", "Nephrology / Urology"),
    ("I feel sad and hopeless and cannot sleep",
     "my mind no dey rest at all i no fit sleep and everything just dey tire me", "Psychiatry"),
    ("I have a toothache and swollen gums",
     "my tooth dey pain me serious and my gum don swell", "Dentistry / Oral Health"),
    ("I have a persistent cough with phlegm",
     "cough no gree leave my chest and catarrh full am", "Pulmonology (Respiratory Medicine)"),
    ("My ear hurts and fluid is coming out",
     "my ear dey pain me and one kind water dey comot from am", "ENT (Otorhinolaryngology)"),
    ("I broke my arm and it is swollen",
     "my hand don break e don swell big", "Orthopaedics"),
    ("I have a bad headache and feel dizzy",
     "my head dey bang me and everywhere dey turn me", "Neurology"),
    ("I have fever, chills and body aches",
     "body dey hot cold dey catch me and all my body dey pain me", "General Medicine"),
]


def margin_of(pipe, text):
    scores = np.sort(np.asarray(pipe.decision_function([text])[0]).ravel())[::-1]
    return float(scores[0] - scores[1]) if scores.size > 1 else float(abs(scores[0]))


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parents[1] / "models" / "triage_models_small.joblib")

    sys.modules["__main__"].clean_medical_text = clean_medical_text
    import joblib
    print(f"loading {model_path} ...")
    models = joblib.load(model_path)
    dept = models["department"]

    formal_ok = pidgin_ok = 0
    print(f"\n{'expected':<38} {'formal pred':<28} {'pidgin pred':<28}")
    print("-" * 100)
    for formal, pidgin, expected in PROBES:
        fp = dept.predict([formal])[0]
        pp = dept.predict([pidgin])[0]
        fm, pm = margin_of(dept, formal), margin_of(dept, pidgin)
        f_ok, p_ok = fp == expected, pp == expected
        formal_ok += f_ok
        pidgin_ok += p_ok
        print(f"{expected[:36]:<38} {('OK ' if f_ok else 'X  ') + fp[:22]:<28} {('OK ' if p_ok else 'X  ') + pp[:22]:<28} m={fm:.2f}/{pm:.2f}")

    n = len(PROBES)
    print(f"\nformal: {formal_ok}/{n}  pidgin: {pidgin_ok}/{n}")


if __name__ == "__main__":
    main()
