"""Lab 2 - Train your first detector and read its confusion matrix.
Run:  python -m labs.lab2_first_model            (threshold 0.5)
      python -m labs.lab2_first_model 0.3        (try other thresholds!)
"""
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from nids.data import load, feature_frame, build_preprocessor

THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

train, test = load("train"), load("test")
Xtr, Xte = feature_frame(train), feature_frame(test)
ytr, yte = train["is_attack"].values, test["is_attack"].values

model = Pipeline([
    ("prep", build_preprocessor()),
    ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)),
])
model.fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:, 1]          # probability that each flow is an attack
pred = (prob >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print(f"Threshold = {THRESHOLD}\n")
print("                 predicted NORMAL   predicted ATTACK")
print(f"actually NORMAL  {tn:>12,}   {fp:>14,}   <- false alarms")
print(f"actually ATTACK  {fn:>12,}   {tp:>14,}")
print("                  ^ missed attacks\n")
print(f"Accuracy           {(tp + tn) / len(yte):.3f}")
print(f"Detection rate     {tp / (tp + fn):.3f}   (share of attacks caught)")
print(f"False alarm rate   {fp / (fp + tn):.3f}   (share of normal traffic flagged)")
print(f"Precision          {tp / max(1, tp + fp):.3f}   (share of alerts that are real)")
