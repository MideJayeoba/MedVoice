#!/usr/bin/env python
"""Evaluate the ORIGINAL 2.5GB triage model on the same test set,
so we can quantify what min_df=3 + ngram(1,2) costs in accuracy."""

import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


def clean_medical_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


sys.modules["__main__"].clean_medical_text = clean_medical_text

import joblib

DOWNLOADS = Path.home() / "Downloads"
ROOT = Path(__file__).resolve().parents[1]

test_df = pd.concat([
    pd.read_csv(DOWNLOADS / "real_world_triage_test.csv"),
    pd.read_csv(DOWNLOADS / "triage_synth_test.csv"),
], ignore_index=True).dropna(subset=["patient_complaint"])

X = test_df["patient_complaint"].fillna("")

print("loading original model (2.5GB)...")
t0 = time.time()
models = joblib.load(ROOT / "models" / "triage_models.joblib")
print(f"loaded in {time.time()-t0:.0f}s")

for name, col in [("category", "disease_category"),
                  ("department", "medical_department"),
                  ("priority", "priority_level")]:
    t0 = time.time()
    pred = models[name].predict(X)
    acc = accuracy_score(test_df[col], pred)
    print(f"{name}: acc={acc*100:.2f}%  ({time.time()-t0:.0f}s)")
