import numpy as np
from fastapi.testclient import TestClient

from nids.data import load, feature_frame, build_preprocessor, ATTACK_CATEGORY, CATEGORIES


def test_dataset_sizes_and_categories():
    tr, te = load("train"), load("test")
    assert len(tr) == 125973 and len(te) == 22544
    assert set(tr["category"]) <= set(CATEGORIES)
    assert set(te["label"]) - set(tr["label"])  # test has unseen attack types


def test_preprocessor_handles_unseen_service():
    tr = feature_frame(load("train").head(500))
    prep = build_preprocessor().fit(tr)
    row = tr.head(1).copy()
    row["service"] = "never_seen_service"
    assert np.isfinite(prep.transform(row)).all()


def test_api_scores_and_explains():
    from nids.api import app
    c = TestClient(app)
    te = load("test")
    flows = feature_frame(te.iloc[:20]).to_dict("records")
    r = c.post("/predict", json={"flows": flows}).json()["results"]
    assert len(r) == 20 and all("reasons" in x and len(x["reasons"]) == 5 for x in r)
    acc = np.mean([x["attack"] == bool(y) for x, y in zip(r, te["is_attack"].iloc[:20])])
    assert acc >= 0.7
