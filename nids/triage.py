"""Triage a batch of network flows the way a SOC analyst works: incidents, not single alerts.

Scoring flows one by one produces thousands of alerts. An analyst needs the batch boiled down
to a handful of incidents, most severe first, with the reasons and a note on what looks new.
This script scores every flow (supervised ensemble + normal-only anomaly detector), groups the
alerts into incidents by (attack category, service, connection flag), and writes:

    results/triage_incidents.csv   one row per incident, ranked
    results/triage_alerts.csv      every alerting flow with its incident id

Usage:
    python -m nids.triage flows.csv            # CSV with the 41 NSL-KDD flow features
    python -m nids.triage --demo               # 3,000 flows sampled from KDDTest+, incl. unseen attack types

Input must use NSL-KDD features (as produced by KDD-style feature extractors). Raw packet
captures or NetFlow/CICFlowMeter output need converting first; see the README.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, load

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [c for c in COLUMNS[:41] if c != "num_outbound_cmds"]
SEVERITY_ORDER = {"high": 0, "medium": 1}
READABLE = {  # SHAP feature names -> what an analyst would call them
    "src_bytes": "bytes sent by the source", "dst_bytes": "bytes sent back", "count": "connections to the same host in 2 s",
    "srv_count": "connections to the same service in 2 s", "serror_rate": "share of SYN errors",
    "srv_serror_rate": "share of SYN errors on the service", "rerror_rate": "share of REJ errors",
    "dst_host_count": "connections to this host (last 100)", "dst_host_srv_count": "connections to this service (last 100)",
    "dst_host_same_srv_rate": "share of the host's traffic on this service",
    "dst_host_diff_srv_rate": "share of the host's traffic spread over services",
    "dst_host_same_src_port_rate": "share from the same source port", "dst_host_serror_rate": "host SYN error rate",
    "dst_host_rerror_rate": "host REJ error rate", "same_srv_rate": "share of connections to the same service",
    "diff_srv_rate": "share of connections to different services", "hot": "'hot' indicators (sensitive files / commands)",
    "num_failed_logins": "failed logins", "logged_in": "logged in", "duration": "connection duration",
    "wrong_fragment": "wrong fragments", "num_compromised": "compromised conditions", "root_shell": "root shell obtained",
}


def _readable(feature: str) -> str:
    for prefix in ("cat__", "log__", "num__"):
        if feature.startswith(prefix):
            feature = feature[len(prefix):]
    for kind in ("service_", "flag_", "protocol_type_"):
        if feature.startswith(kind):
            return f"{kind[:-1].replace('_type', '')} = {feature[len(kind):]}"
    return READABLE.get(feature, feature.replace("_", " "))


def triage(flows: pd.DataFrame):
    from .api import score     # loads the trained model bundle
    X = flows.copy()
    if "num_outbound_cmds" not in X:
        X["num_outbound_cmds"] = 0
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise SystemExit(f"Input is missing NSL-KDD features, e.g. {missing[:5]}")
    X = X[COLUMNS[:41]].drop(columns=["num_outbound_cmds"])
    res = pd.DataFrame(score(X, explain=False))
    alerts = flows.loc[res["attack"].values].copy()
    alerts[["probability", "category", "zero_day_anomaly", "severity"]] = res.loc[res["attack"],
        ["probability", "category", "zero_day_anomaly", "severity"]].values
    if alerts.empty:
        return alerts, pd.DataFrame()

    # explain only alerts (SHAP is the slow part)
    from .api import explainer
    reasons = explainer.reasons(X.loc[alerts.index], top=3)
    alerts["top_reasons"] = ["; ".join(_readable(r["feature"]) for r in rs if r["direction"] == "towards ATTACK")
                             or "unusual compared with normal traffic (anomaly detector)" for rs in reasons]

    alerts["incident_key"] = alerts["category"] + " | " + alerts["service"] + " | " + alerts["flag"]
    rows = []
    for key, g in alerts.groupby("incident_key"):
        reasons_count = Counter(r for rs in g["top_reasons"] for r in rs.split("; "))
        rows.append({
            "category": g["category"].iloc[0], "service": g["service"].iloc[0], "flag": g["flag"].iloc[0],
            "flows": len(g), "high_severity": int((g["severity"] == "high").sum()),
            "anomaly_flagged_flows": int(g["zero_day_anomaly"].astype(bool).sum()),   # unlike normal traffic
            "max_probability": round(float(g["probability"].astype(float).max()), 3),
            "severity": "high" if (g["severity"] == "high").any() else "medium",
            "main_reasons": "; ".join(r for r, _ in reasons_count.most_common(3)),
        })
    inc = pd.DataFrame(rows)
    inc["rank_key"] = inc["severity"].map(SEVERITY_ORDER) * 1e9 - inc["flows"]
    inc = inc.sort_values("rank_key").drop(columns="rank_key").reset_index(drop=True)
    inc.insert(0, "incident", [f"INC-{i + 1:03d}" for i in range(len(inc))])
    alerts["incident"] = alerts["incident_key"].map(dict(zip(inc["category"] + " | " + inc["service"] + " | " + inc["flag"],
                                                             inc["incident"])))
    return alerts.drop(columns="incident_key"), inc


def main():
    ap = argparse.ArgumentParser(description="Turn a batch of flows into a ranked incident list.")
    ap.add_argument("flows_csv", nargs="?")
    ap.add_argument("--demo", action="store_true", help="use 3,000 flows sampled from KDDTest+")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    if a.demo:
        test, seen = load("test"), set(load("train")["label"])
        flows = test.sample(3000, random_state=7).reset_index(drop=True)
        flows["unseen_type_hindsight"] = ~flows["label"].isin(seen)
    elif a.flows_csv:
        flows = pd.read_csv(a.flows_csv)
    else:
        ap.error("give a flows CSV or --demo")
    alerts, inc = triage(flows)
    a.out.mkdir(exist_ok=True)
    inc.to_csv(a.out / "triage_incidents.csv", index=False)
    alerts.to_csv(a.out / "triage_alerts.csv", index=False)
    print(f"{len(flows):,} flows -> {len(alerts):,} alerting flows -> {len(inc)} incidents "
          f"({(inc['severity'] == 'high').sum()} high severity)")
    if "is_attack" in flows:
        attacks = flows["is_attack"] == 1
        covered = flows.index.isin(alerts.index)
        print(f"hindsight: {covered[attacks].mean():.0%} of real attack flows are inside an incident; "
              f"{(~flows.loc[alerts.index, 'is_attack'].astype(bool)).mean():.0%} of alerting flows were normal traffic")
        if "unseen_type_hindsight" in flows:
            u = flows["unseen_type_hindsight"] & attacks
            print(f"           {covered[u].mean():.0%} of flows from attack types never seen in training were caught")
    print(inc.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
