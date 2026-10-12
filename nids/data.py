"""NSL-KDD loading and preprocessing.

NSL-KDD (Tavallaee et al., 2009) fixes the duplicate-record problems of
KDD'99. KDDTest+ contains 17 attack types that never appear in training –
a realistic test of generalisation to new attacks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

DATA = Path(__file__).resolve().parents[1] / "data"
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root",
                "num_file_creations", "num_access_files", "count", "srv_count",
                "dst_host_count", "dst_host_srv_count"]

ATTACK_CATEGORY = {
    "normal": "normal",
    # Denial of Service
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
                          "processtable", "udpstorm"]},
    # Probing / reconnaissance
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    # Remote to Local
    **{a: "R2L" for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
                          "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock",
                          "xsnoop", "worm"]},
    # User to Root (privilege escalation)
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps",
                          "sqlattack", "xterm"]},
}
CATEGORIES = ["normal", "DoS", "Probe", "R2L", "U2R"]


def load(split: str = "train") -> pd.DataFrame:
    f = DATA / ("KDDTrain+.txt.gz" if split == "train" else "KDDTest+.txt.gz")
    df = pd.read_csv(f, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY).fillna("R2L")
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[COLUMNS[:41]].drop(columns=["num_outbound_cmds"])  # constant zero column


def build_preprocessor() -> ColumnTransformer:
    numeric = [c for c in COLUMNS[:41] if c not in CATEGORICAL + ["num_outbound_cmds"]]
    heavy = [c for c in numeric if c in HEAVY_TAILED]
    light = [c for c in numeric if c not in HEAVY_TAILED]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("sc", StandardScaler())]), heavy),
        ("num", StandardScaler(), light),
    ], verbose_feature_names_out=False)
