"""Train & evaluate.

1. Binary attack detection and 5-class categorisation with
   Logistic Regression, Random Forest, XGBoost and a soft-voting ensemble.
2. Isolation Forest trained ONLY on normal traffic (zero-day detector).
3. Hybrid = ensemble OR anomaly detector (catches attacks never seen in training).
Evaluated on the official KDDTest+ and on attack types unseen in training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import load, feature_frame, build_preprocessor, CATEGORIES

ROOT = Path(__file__).resolve().parents[1]


def binary_metrics(y, pred, score=None):
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
         "recall(detection_rate)": recall_score(y, pred), "f1": f1_score(y, pred),
         "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))}
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score)
    return {k: round(float(v), 4) for k, v in m.items()}


def models(n_classes=2):
    obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1,
                                               random_state=0, class_weight="balanced_subsample"),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.8, objective=obj, tree_method="hist", n_jobs=2,
                                 random_state=0, eval_metric="logloss"),
    }


def main():
    tr, te = load("train"), load("test")
    Xtr, Xte = feature_frame(tr), feature_frame(te)
    seen = set(tr["label"])
    unseen_mask = ~te["label"].isin(seen)
    out = {"dataset": {"train_rows": len(tr), "test_rows": len(te),
                       "test_attack_types_unseen_in_train": sorted(set(te["label"]) - seen),
                       "test_rows_with_unseen_attacks": int(unseen_mask.sum())}}

    # ---------------- binary ----------------
    ytr, yte = tr["is_attack"].values, te["is_attack"].values
    res, fitted = {}, {}
    for name, est in models(2).items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", est)]).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        res[name] = {**binary_metrics(yte, (p >= 0.5).astype(int), p), "train_s": round(time.time() - t0, 1)}
        fitted[name] = pipe
        print(name, res[name], flush=True)
    ens = Pipeline([("prep", build_preprocessor()),
                    ("clf", VotingClassifier([(k, v) for k, v in models(2).items()], voting="soft",
                                             weights=[1, 2, 2]))]).fit(Xtr, ytr)
    pe = ens.predict_proba(Xte)[:, 1]
    res["SoftVotingEnsemble"] = binary_metrics(yte, (pe >= 0.5).astype(int), pe)
    print("ensemble", res["SoftVotingEnsemble"], flush=True)

    # ---------------- anomaly detector (normal traffic only) ----------------
    prep = build_preprocessor().fit(Xtr)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0, n_jobs=-1)
    iso.fit(prep.transform(Xtr[ytr == 0]))
    a_score = -iso.score_samples(prep.transform(Xte))
    # threshold = 99th percentile of anomaly score on NORMAL training traffic (≈1% FPR budget)
    thr = float(np.quantile(-iso.score_samples(prep.transform(Xtr[ytr == 0])), 0.99))
    a_pred = (a_score >= thr).astype(int)
    res["IsolationForest(normal-only)"] = binary_metrics(yte, a_pred, a_score)

    hybrid = ((pe >= 0.5) | (a_pred == 1)).astype(int)
    res["Hybrid(Ensemble OR IsolationForest)"] = binary_metrics(yte, hybrid)
    out["binary_KDDTest+"] = res
    out["detection_rate_on_unseen_attack_types"] = {
        "SoftVotingEnsemble": round(float((pe[unseen_mask] >= 0.5).mean()), 4),
        "IsolationForest": round(float(a_pred[unseen_mask].mean()), 4),
        "Hybrid": round(float(hybrid[unseen_mask].mean()), 4)}

    # ---------------- 5-class ----------------
    cat_to_i = {c: i for i, c in enumerate(CATEGORIES)}
    ctr, cte = tr["category"].map(cat_to_i).values, te["category"].map(cat_to_i).values
    mc = Pipeline([("prep", build_preprocessor()), ("clf", models(5)["XGBoost"])]).fit(Xtr, ctr)
    pc = mc.predict(Xte)
    rep = classification_report(cte, pc, target_names=CATEGORIES, output_dict=True, zero_division=0)
    out["multiclass_XGBoost_KDDTest+"] = {
        "accuracy": round(rep["accuracy"], 4), "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "per_class_recall": {c: round(rep[c]["recall"], 4) for c in CATEGORIES},
        "confusion_matrix": confusion_matrix(cte, pc).tolist()}

    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2))
    joblib.dump({"binary": ens, "multiclass": mc, "iso": iso, "iso_prep": prep, "iso_threshold": thr,
                 "categories": CATEGORIES}, ROOT / "models" / "nids.joblib", compress=3)
    print(json.dumps({k: out[k] for k in ["binary_KDDTest+", "detection_rate_on_unseen_attack_types"]}, indent=2))
    print(json.dumps(out["multiclass_XGBoost_KDDTest+"], indent=2))


if __name__ == "__main__":
    main()
