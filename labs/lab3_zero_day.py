"""Lab 3 - The zero-day problem: what happens to attacks the model has never seen?
Run:  python -m labs.lab3_zero_day
      python -m labs.lab3_zero_day 97      (change the anomaly threshold percentile)
"""
import sys

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline

from nids.data import load, feature_frame, build_preprocessor

PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 99.0

train, test = load("train"), load("test")
Xtr, Xte = feature_frame(train), feature_frame(test)
ytr, yte = train["is_attack"].values, test["is_attack"].values
unseen = ~test["label"].isin(set(train["label"]))
seen_attack = (yte == 1) & ~unseen

rf = Pipeline([("prep", build_preprocessor()), ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0))])
rf.fit(Xtr, ytr)
sup = rf.predict(Xte)

prep = build_preprocessor().fit(Xtr)
iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(prep.transform(Xtr[ytr == 0]))   # NORMAL only!
normal_scores = -iso.score_samples(prep.transform(Xtr[ytr == 0]))
cut = np.percentile(normal_scores, PCT)
anom = (-iso.score_samples(prep.transform(Xte)) >= cut).astype(int)
hybrid = sup | anom

def rate(mask, pred):
    return f"{pred[mask].mean():6.1%}"

print(f"Anomaly threshold = {PCT}th percentile of normal training traffic\n")
print(f"{'':28}{'Random Forest':>14}{'Anomaly':>10}{'Hybrid':>10}")
print(f"{'Known attack types caught':28}{rate(seen_attack, sup):>14}{rate(seen_attack, anom):>10}{rate(seen_attack, hybrid):>10}")
print(f"{'NEW attack types caught':28}{rate(unseen, sup):>14}{rate(unseen, anom):>10}{rate(unseen, hybrid):>10}")
print(f"{'False alarms on normal':28}{rate(yte == 0, sup):>14}{rate(yte == 0, anom):>10}{rate(yte == 0, hybrid):>10}")

print("\nNew attack types, one by one (hybrid detection rate):")
t = test[unseen].assign(caught=hybrid[unseen.values])
print(t.groupby("label")["caught"].agg(["count", "mean"]).sort_values("count", ascending=False)
      .rename(columns={"count": "flows", "mean": "caught"}).round(2).head(10).to_string())
