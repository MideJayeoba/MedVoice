#!/usr/bin/env python
"""Train the production triage v2 bundle.

v2 architecture (winners from scripts/compare_models2.py):
  category : LinearSVC on the 68-class patient-comment dataset (88.1% test)
  priority : LinearSVC on the large complaint dataset (74.4% test)
  department is NOT a model anymore — it comes from
  data/category_to_department.json at inference time.

Saves models/triage_models_v2.joblib (models/ is gitignored — push manually
when approved).
"""

import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
OUT = ROOT / "models" / "triage_models_v2.joblib"

CANON = {
    "growth issue": "Growth issue",
    "mood swing": "Mood swing",
    "Pregnancy issues": "pregnancy issues",
    "Unexplained Fever/ Bruising": "Unexplained Fever/Bruising",
}


def clean_medical_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def main():
    # ---- category (68 classes, new dataset) --------------------------------
    tr = pd.read_excel(ROOT / "data/raw/disease/HealthCare Data.xlsx")
    te = pd.read_excel(ROOT / "data/raw/disease/Test_data.xlsx")
    for df in (tr, te):
        df["Patient_Category"] = df["Patient_Category"].replace(CANON)
    te = te[~te["Patient_comment"].isin(set(tr["Patient_comment"]))]

    t0 = time.time()
    cat_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_medical_text, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto")),
    ])
    cat_pipe.fit(tr["Patient_comment"].astype(str), tr["Patient_Category"])
    acc = accuracy_score(te["Patient_Category"], cat_pipe.predict(te["Patient_comment"].astype(str)))
    print(f"category68: acc={acc*100:.2f}% ({time.time()-t0:.0f}s, {len(tr)} train rows)")

    # ---- priority (existing large dataset) ---------------------------------
    frames = [pd.read_csv(f) for f in [
        DOWNLOADS / "real_world_triage_train.csv",
        DOWNLOADS / "triage_synth_train.csv",
        DOWNLOADS / "triage_everyday_train.csv",
    ] if f.exists()]
    big = pd.concat(frames, ignore_index=True).dropna(subset=["patient_complaint"])
    te_frames = [pd.read_csv(f) for f in [
        DOWNLOADS / "real_world_triage_test.csv",
        DOWNLOADS / "triage_synth_test.csv",
        DOWNLOADS / "triage_everyday_test.csv",
    ] if f.exists()]
    big_te = pd.concat(te_frames, ignore_index=True).dropna(subset=["patient_complaint"])

    t0 = time.time()
    prio_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_medical_text, stop_words="english",
                                  ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
        ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto")),
    ])
    prio_pipe.fit(big["patient_complaint"].fillna(""), big["priority_level"])
    acc = accuracy_score(big_te["priority_level"], prio_pipe.predict(big_te["patient_complaint"].fillna("")))
    print(f"priority:   acc={acc*100:.2f}% ({time.time()-t0:.0f}s, {len(big)} train rows)")

    joblib.dump({"category": cat_pipe, "priority": prio_pipe, "version": 2}, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
