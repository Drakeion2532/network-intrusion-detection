"""Lab 5 - Talk to the detector like a SOC tool would.
Start the API in another terminal first:   uvicorn nids.api:app --port 8001
Run:  python -m labs.lab5_api_client
"""
import json
import urllib.request

from nids.data import load, feature_frame

flow = feature_frame(load("test").query("label == 'neptune'").head(1)).iloc[0].to_dict()
flow = {k: (v.item() if hasattr(v, "item") else v) for k, v in flow.items()}

def ask(f):
    req = urllib.request.Request("http://127.0.0.1:8001/predict", data=json.dumps({"flows": [f]}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["results"][0]

r = ask(flow)
print("Original SYN-flood flow ->", r["category"], f"p={r['probability']}", "zero-day flag:", r["zero_day_anomaly"])

# Experiment: make the flood look like a normal web request. Which changes flip the verdict?
disguised = dict(flow, service="http", flag="SF", src_bytes=230, dst_bytes=2000, logged_in=1,
                 serror_rate=0.0, srv_serror_rate=0.0, dst_host_serror_rate=0.0, dst_host_srv_serror_rate=0.0)
r2 = ask(disguised)
print("Disguised flow          ->", r2["category"], f"p={r2['probability']}", "zero-day flag:", r2["zero_day_anomaly"])
print("Top reasons now:", [x["feature"] for x in r2["reasons"]])
