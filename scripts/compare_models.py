#!/usr/bin/env python
"""Compare classifier algorithms for the triage task.

Trains each algorithm on the same TF-IDF features and test split used by
retrain_eval.py, for all three targets (category / department / priority),
and prints an accuracy + macro-F1 + timing comparison table.

Algorithms:
  - LinearSVC            (current production model)
  - LogisticRegression   (probabilistic — would give real confidences)
  - SGDClassifier        (hinge loss, scales well, supports partial_fit)
  - MultinomialNB        (fast classic text baseline)
  - ComplementNB         (NB variant designed for imbalanced text classes)

Usage: python scripts/compare_models.py [--quick]
  --quick uses a 30k-row training sample for a fast first pass.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

DOWNLOADS = Path.home() / "Downloads"
TRAIN_FILES = [
    DOWNLOADS / "real_world_triage_train.csv",
    DOWNLOADS / "triage_synth_train.csv",
    DOWNLOADS / "triage_everyday_train.csv",
]
TEST_FILES = [
    DOWNLOADS / "real_world_triage_test.csv",
    DOWNLOADS / "triage_synth_test.csv",
    DOWNLOADS / "triage_everyday_test.csv",
]

TARGETS = {
    "category": "disease_category",
    "department": "medical_department",
    "priority": "priority_level",
}

ALGOS = {
    "LinearSVC": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto"),
    "LogisticRegression": lambda: LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=2000, random_state=42, n_jobs=-1),
    "SGD-hinge": lambda: SGDClassifier(
        loss="hinge", alpha=1e-5, class_weight="balanced", random_state=42, n_jobs=-1),
    "MultinomialNB": lambda: MultinomialNB(alpha=0.3),
    "ComplementNB": lambda: ComplementNB(alpha=0.3),
}


def clean_medical_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def load(files):
    frames = [pd.read_csv(f) for f in files if f.exists()]
    return pd.concat(frames, ignore_index=True).dropna(subset=["patient_complaint"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="30k training sample for a fast pass")
    args = ap.parse_args()

    train_df = load(TRAIN_FILES)
    test_df = load(TEST_FILES)
    if args.quick:
        train_df = train_df.sample(30_000, random_state=42)
    print(f"train rows: {len(train_df)}  test rows: {len(test_df)}\n")

    # Vectorise ONCE — same features for every algorithm, fair comparison
    print("fitting TF-IDF ...")
    t0 = time.time()
    vec = TfidfVectorizer(preprocessor=clean_medical_text, stop_words="english",
                          ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train = vec.fit_transform(train_df["patient_complaint"].fillna(""))
    X_test = vec.transform(test_df["patient_complaint"].fillna(""))
    print(f"TF-IDF done: {X_train.shape[1]:,} features ({time.time()-t0:.0f}s)\n")

    results = []
    for target, col in TARGETS.items():
        y_train, y_test = train_df[col], test_df[col]
        for name, make in ALGOS.items():
            clf = make()
            t0 = time.time()
            try:
                clf.fit(X_train, y_train)
                fit_s = time.time() - t0
                pred = clf.predict(X_test)
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                results.append((target, name, acc, f1, fit_s))
                print(f"  {target:10} {name:20} acc={acc*100:6.2f}%  macroF1={f1:.3f}  fit={fit_s:5.0f}s")
            except Exception as exc:
                print(f"  {target:10} {name:20} FAILED: {exc}")

    print("\n=== SUMMARY (best per target by accuracy) ===")
    df = pd.DataFrame(results, columns=["target", "algo", "acc", "macro_f1", "fit_s"])
    for target in TARGETS:
        sub = df[df.target == target].sort_values("acc", ascending=False)
        best = sub.iloc[0]
        print(f"\n{target}: winner = {best.algo} ({best.acc*100:.2f}%)")
        for _, r in sub.iterrows():
            print(f"    {r.algo:20} acc={r.acc*100:6.2f}%  macroF1={r.macro_f1:.3f}  fit={r.fit_s:5.0f}s")

    out = Path(__file__).resolve().parents[1] / "models" / "algo_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved results -> {out}")


if __name__ == "__main__":
    main()
