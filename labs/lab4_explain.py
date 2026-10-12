"""Lab 4 - Why did the model raise this alert? (SHAP)
Run:  python -m labs.lab4_explain                 (a SYN flood)
      python -m labs.lab4_explain guess_passwd    (any attack name from Lab 1)
Needs the trained model: run  python -m nids.train  once first.
"""
import sys

import joblib

from nids.data import load, feature_frame
from nids.explain import Explainer

attack = sys.argv[1] if len(sys.argv) > 1 else "neptune"
bundle = joblib.load("models/nids.joblib")
ex = Explainer(bundle)
test = load("test")
rows = test[test["label"] == attack]
if rows.empty:
    sys.exit(f"No test flows labelled {attack!r}. Pick a name printed by Lab 1.")
flow = feature_frame(rows.head(1))
p = bundle["binary"].predict_proba(flow)[0, 1]
print(f"Flow type: {attack}   model's attack probability: {p:.3f}\n")
print("Top reasons (positive = pushed towards ATTACK):")
for r in ex.reasons(flow, top=6)[0]:
    print(f"  {r['feature']:<32} {r['impact']:+.3f}")
print("\nThe raw values behind those features:")
cols = []
for r in ex.reasons(flow, top=6)[0]:
    f = r["feature"]
    raw = f if f in flow.columns else next(c for c in ("protocol_type", "service", "flag") if f.startswith(c + "_"))
    if raw not in cols:
        cols.append(raw)
print(flow[cols].to_string(index=False))
