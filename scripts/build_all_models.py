#!/usr/bin/env python
"""Train and SAVE one loadable bundle per algorithm from compare_models.py.

Each bundle is {vectorizer, category, department, priority} where the
vectorizer is shared (fit once) and each target has its own classifier.
Saved to models/compare/triage_<algo>.joblib — models/ is gitignored,
so these stay local for offline comparison and are never pushed.

Usage: python scripts/build_all_models.py [--algos LinearSVC SGD-hinge ...]
"""

import argparse
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

DOWNLOADS = Path.home() / "Downloads"
TRAIN_FILES = [
    DOWNLOADS / "real_world_triage_train.csv",
    DOWNLOADS / "triage_synth_train.csv",
    DOWNLOADS / "triage_everyday_train.csv",
]
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "compare"

TARGETS = {
    "category": "disease_category",
    "department": "medical_department",
    "priority": "priority_level",
}

ALGOS = {
    "LinearSVC": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto"),
    "LogisticRegression": lambda: LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=2000, random_state=42),
    "SGD-hinge": lambda: SGDClassifier(
        loss="hinge", alpha=1e-5, class_weight="balanced", random_state=42),
    "MultinomialNB": lambda: MultinomialNB(alpha=0.3),
    "ComplementNB": lambda: ComplementNB(alpha=0.3),
}


def clean_medical_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=list(ALGOS), choices=list(ALGOS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat(
        [pd.read_csv(f) for f in TRAIN_FILES if f.exists()], ignore_index=True
    ).dropna(subset=["patient_complaint"])
    print(f"train rows: {len(train_df)}")

    print("fitting shared TF-IDF ...")
    t0 = time.time()
    vec = TfidfVectorizer(preprocessor=clean_medical_text, stop_words="english",
                          ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(train_df["patient_complaint"].fillna(""))
    print(f"TF-IDF: {X.shape[1]:,} features ({time.time()-t0:.0f}s)")

    for name in args.algos:
        bundle = {"vectorizer": vec, "algo": name}
        for target, col in TARGETS.items():
            t0 = time.time()
            clf = ALGOS[name]()
            clf.fit(X, train_df[col])
            bundle[target] = clf
            print(f"  {name:20} {target:10} fit={time.time()-t0:5.0f}s")
        path = OUT_DIR / f"triage_{name.lower().replace('-', '_')}.joblib"
        joblib.dump(bundle, path, compress=3)
        print(f"  -> saved {path} ({path.stat().st_size/1e6:.0f} MB)\n")

    print("All bundles saved. Load with:")
    print("  b = joblib.load(path); X = b['vectorizer'].transform([text])")
    print("  b['department'].predict(X)")


if __name__ == "__main__":
    main()
