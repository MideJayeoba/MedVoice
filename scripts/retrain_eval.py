#!/usr/bin/env python
"""Retrain compact triage models + detailed error analysis.

Trains with min_df=3, ngram (1,2) — much smaller than the original
min_df=1, ngram (1,3) — and reports where accuracy suffers so we can
generate targeted augmentation data.

Usage: python scripts/retrain_eval.py [--augment path.csv ...] [--save]
"""

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DOWNLOADS = Path.home() / "Downloads"
TRAIN_FILES = [DOWNLOADS / "real_world_triage_train.csv", DOWNLOADS / "triage_synth_train.csv"]
TEST_FILES = [DOWNLOADS / "real_world_triage_test.csv", DOWNLOADS / "triage_synth_test.csv"]

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "models" / "triage_models_small.joblib"

TARGETS = {
    "category": "disease_category",
    "department": "medical_department",
    "priority": "priority_level",
}


def clean_medical_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def make_pipeline(min_df: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_medical_text,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto")),
    ])


def load(files):
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["patient_complaint"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", nargs="*", default=[], help="extra training CSVs")
    ap.add_argument("--save", action="store_true", help="save bundled model")
    args = ap.parse_args()

    train_df = load(TRAIN_FILES + [Path(p) for p in args.augment])
    test_df = load(TEST_FILES)
    print(f"train rows: {len(train_df)}  test rows: {len(test_df)}")
    if args.augment:
        print(f"  (including augmentation: {args.augment})")

    X_train = train_df["patient_complaint"].fillna("")
    X_test = test_df["patient_complaint"].fillna("")

    bundle = {}
    for name, col in TARGETS.items():
        t0 = time.time()
        pipe = make_pipeline(min_df=3)
        pipe.fit(X_train, train_df[col])
        pred = pipe.predict(X_test)
        acc = accuracy_score(test_df[col], pred)
        vocab = len(pipe.named_steps["tfidf"].vocabulary_)
        print(f"\n=== {name}: acc={acc*100:.2f}%  vocab={vocab:,}  ({time.time()-t0:.0f}s) ===")

        # Per-class report — only show weak classes (f1 < 0.85)
        report = classification_report(test_df[col], pred, output_dict=True, zero_division=0)
        weak = {k: v for k, v in report.items()
                if isinstance(v, dict) and "f1-score" in v
                and k not in ("macro avg", "weighted avg") and v["f1-score"] < 0.85}
        if weak:
            print("  weak classes (f1 < 0.85):")
            for cls, m in sorted(weak.items(), key=lambda kv: kv[1]["f1-score"]):
                print(f"    {cls[:45]:45} f1={m['f1-score']:.2f} precision={m['precision']:.2f} recall={m['recall']:.2f} n={int(m['support'])}")

        # Top confusion pairs
        pairs = Counter(
            (t, p) for t, p in zip(test_df[col], pred) if t != p
        )
        print("  top confusions (true -> predicted):")
        for (t, p), n in pairs.most_common(8):
            print(f"    {n:5}  {t[:38]:38} -> {p[:38]}")

        bundle[name] = pipe

    if args.save:
        OUT_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, OUT_PATH, compress=3)
        size_mb = OUT_PATH.stat().st_size / 1e6
        print(f"\nSaved {OUT_PATH} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
