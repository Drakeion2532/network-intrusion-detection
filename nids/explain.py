"""SHAP explanations: global feature importance plot + per-flow reasons."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class Explainer:
    def __init__(self, bundle):
        import shap
        pipe = bundle["binary"]
        self.prep = pipe.named_steps["prep"]
        self.xgb = pipe.named_steps["clf"].named_estimators_["XGBoost"]
        self.names = list(self.prep.get_feature_names_out())
        self.shap = shap.TreeExplainer(self.xgb)

    def reasons(self, X: pd.DataFrame, top=5):
        Z = self.prep.transform(X)
        sv = self.shap.shap_values(Z)
        out = []
        for row in np.atleast_2d(sv):
            idx = np.argsort(-np.abs(row))[:top]
            out.append([{"feature": self.names[i], "impact": round(float(row[i]), 3),
                         "direction": "towards ATTACK" if row[i] > 0 else "towards normal"} for i in idx])
        return out


def global_importance_plot(n=4000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from .data import load, feature_frame
    bundle = joblib.load(ROOT / "models" / "nids.joblib")
    ex = Explainer(bundle)
    X = feature_frame(load("test").sample(n, random_state=0))
    Z = ex.prep.transform(X)
    sv = ex.shap.shap_values(Z)
    shap.summary_plot(sv, Z, feature_names=ex.names, max_display=15, show=False)
    plt.title("What drives the attack prediction (SHAP, XGBoost member)")
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / "shap_summary.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    global_importance_plot()
