"""REST API for scoring network flows.

    uvicorn nids.api:app --port 8001
    POST /predict   {"flows": [ {<41 NSL-KDD features>}, ... ]}
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .data import COLUMNS
from .explain import Explainer

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "models" / "nids.joblib")
explainer = Explainer(bundle)
FEATURES = [c for c in COLUMNS[:41] if c != "num_outbound_cmds"]
app = FastAPI(title="Network Intrusion Detection API")


class Batch(BaseModel):
    flows: list[dict]
    explain: bool = True


def score(df: pd.DataFrame, explain=True):
    p = bundle["binary"].predict_proba(df)[:, 1]
    cats = bundle["multiclass"].predict(df)
    iso = -bundle["iso"].score_samples(bundle["iso_prep"].transform(df))
    reasons = explainer.reasons(df) if explain else [None] * len(df)
    res = []
    for i in range(len(df)):
        anomalous = bool(iso[i] >= bundle["iso_threshold"])
        attack = bool(p[i] >= 0.5 or anomalous)
        sev = "high" if p[i] >= 0.9 else "medium" if attack else "none"
        res.append({"attack": attack, "probability": round(float(p[i]), 4),
                    "category": bundle["categories"][int(cats[i])] if p[i] >= 0.5 else
                    ("unknown-anomaly" if anomalous else "normal"),
                    "zero_day_anomaly": anomalous, "severity": sev, "reasons": reasons[i]})
    return res


@app.post("/predict")
def predict(b: Batch):
    df = pd.DataFrame(b.flows)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing features: {missing[:5]}...")
    df["num_outbound_cmds"] = 0
    return {"results": score(df[COLUMNS[:41]].drop(columns=["num_outbound_cmds"]), b.explain)}


@app.get("/health")
def health():
    return {"status": "ok", "model": "SoftVoting(LR,RF,XGB) + IsolationForest"}
