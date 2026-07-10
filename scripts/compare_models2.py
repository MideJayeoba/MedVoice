#!/usr/bin/env python
"""Algorithm comparison v2 — for the new triage architecture.

Tasks:
  category68 : Patient_comment -> 68 fine categories (new dataset, TF-IDF text)
  priority   : complaint text  -> Emergency/High/Moderate/Low (existing data)
  disease    : symptoms+age+gender -> 30 diseases (Healthcare.csv — included
               to demonstrate whether it carries any signal)

Algorithms: LinearSVC, LogisticRegression, SGD-hinge, MultinomialNB,
            ComplementNB, RandomForest, KNN.

Notes on fairness:
  - Text tasks: linear/NB models use full sparse TF-IDF. RandomForest and
    KNN cannot handle 250k sparse dims well, so they get TruncatedSVD-300
    of the same TF-IDF (standard practice); NB needs non-negative input so
    it always uses raw TF-IDF.
  - disease uses a small dense matrix (28 symptom flags + age + gender),
    every algorithm sees identical features.

Output: table per task + models/algo_comparison2.csv. Nothing is pushed.
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"

# canonicalize label variants to the doctor_types mapping keys
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


def algos(seed=42):
    return {
        "LinearSVC": LinearSVC(C=0.5, class_weight="balanced", random_state=seed, dual="auto"),
        "LogisticRegression": LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000, random_state=seed),
        "SGD-hinge": SGDClassifier(loss="hinge", alpha=1e-5, class_weight="balanced", random_state=seed),
        "MultinomialNB": MultinomialNB(alpha=0.3),
        "ComplementNB": ComplementNB(alpha=0.3),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
    }

NEEDS_DENSE = {"RandomForest", "KNN"}
NEEDS_NONNEG = {"MultinomialNB", "ComplementNB"}

results = []


def run_task(task, Xtr_sparse, Xte_sparse, ytr, yte, svd_dims=300):
    # shared SVD projection for dense-only algorithms
    Xtr_svd = Xte_svd = None
    for name, clf in algos().items():
        if name in NEEDS_DENSE and Xtr_sparse.shape[1] > 2000:
            if Xtr_svd is None:
                svd = TruncatedSVD(n_components=svd_dims, random_state=42)
                Xtr_svd = svd.fit_transform(Xtr_sparse)
                Xte_svd = svd.transform(Xte_sparse)
            Xtr, Xte = Xtr_svd, Xte_svd
            feats = f"SVD-{svd_dims}"
        else:
            Xtr, Xte = Xtr_sparse, Xte_sparse
            feats = "full"
        if name in NEEDS_NONNEG:
            Xtr, Xte, feats = Xtr_sparse, Xte_sparse, "full"
        t0 = time.time()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        dt = time.time() - t0
        print(f"  {task:10} {name:20} acc={acc*100:6.2f}%  macroF1={f1:.3f}  fit={dt:5.0f}s  [{feats}]", flush=True)
        results.append({"task": task, "algo": name, "accuracy": round(acc, 4),
                        "macro_f1": round(f1, 4), "fit_seconds": round(dt, 1), "features": feats})


# ---------------------------------------------------------------- category68
print("== category68 (new comment dataset) ==", flush=True)
tr = pd.read_excel(ROOT / "data/raw/disease/HealthCare Data.xlsx")
te = pd.read_excel(ROOT / "data/raw/disease/Test_data.xlsx")
for df in (tr, te):
    df["Patient_Category"] = df["Patient_Category"].replace(CANON)
# drop test rows whose exact comment leaked into train
te = te[~te["Patient_comment"].isin(set(tr["Patient_comment"]))]
vec = TfidfVectorizer(preprocessor=clean_medical_text, stop_words="english",
                      ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(tr["Patient_comment"].astype(str))
Xte = vec.transform(te["Patient_comment"].astype(str))
print(f"train {Xtr.shape[0]} rows, test {Xte.shape[0]} rows, {Xtr.shape[1]:,} features", flush=True)
run_task("category68", Xtr, Xte, tr["Patient_Category"], te["Patient_Category"])

# ---------------------------------------------------------------- priority
print("\n== priority (existing dataset) ==", flush=True)
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
vec_p = TfidfVectorizer(preprocessor=clean_medical_text, stop_words="english",
                        ngram_range=(1, 2), min_df=3, sublinear_tf=True)
Xtr = vec_p.fit_transform(big["patient_complaint"].fillna(""))
Xte = vec_p.transform(big_te["patient_complaint"].fillna(""))
print(f"train {Xtr.shape[0]} rows, test {Xte.shape[0]} rows, {Xtr.shape[1]:,} features", flush=True)
run_task("priority", Xtr, Xte, big["priority_level"], big_te["priority_level"])

# ---------------------------------------------------------------- disease
print("\n== disease (Healthcare.csv — signal check) ==", flush=True)
d = pd.read_csv(ROOT / "data/raw/category/Healthcare.csv")
mlb = MultiLabelBinarizer()
X_sym = mlb.fit_transform(d["Symptoms"].str.split(", "))
X = np.hstack([X_sym, (d[["Age"]].values / 90.0), pd.get_dummies(d["Gender"]).values]).astype(np.float32)
from scipy import sparse
Xs = sparse.csr_matrix(X)
Xtr, Xte, ytr, yte = train_test_split(Xs, d["Disease"], test_size=0.2, random_state=42, stratify=d["Disease"])
print(f"train {Xtr.shape[0]} rows, test {Xte.shape[0]} rows, {Xtr.shape[1]} features "
      f"(random-guess baseline = {100/30:.1f}%)", flush=True)
run_task("disease", Xtr, Xte, ytr, yte)

# ---------------------------------------------------------------- summary
print("\n=== SUMMARY (best per task) ===")
df = pd.DataFrame(results)
for task, grp in df.groupby("task"):
    best = grp.sort_values("accuracy", ascending=False).iloc[0]
    print(f"{task}: winner = {best.algo} ({best.accuracy*100:.2f}%)")
out = ROOT / "models" / "algo_comparison2.csv"
df.to_csv(out, index=False)
print(f"saved -> {out}")
