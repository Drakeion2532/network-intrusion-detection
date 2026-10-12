"""Lab 1 - Meet the data.
Run:  python -m labs.lab1_explore
"""
import pandas as pd

from nids.data import load, CATEGORIES

train, test = load("train"), load("test")
print(f"Training flows: {len(train):,}   Test flows: {len(test):,}")
print(f"Columns per flow: {train.shape[1] - 4} features + label, difficulty, category, is_attack\n")

print("1) How many flows of each category?")
summary = pd.DataFrame({
    "train": train["category"].value_counts(),
    "test": test["category"].value_counts(),
}).reindex(CATEGORIES)
print(summary.to_string(), "\n")

print("2) Top 8 attack names in TRAINING")
print(train.loc[train.is_attack == 1, "label"].value_counts().head(8).to_string(), "\n")

unseen = sorted(set(test["label"]) - set(train["label"]))
print(f"3) Attack types that appear ONLY in the test set ({len(unseen)}):")
print(", ".join(unseen))
print(f"   -> {test['label'].isin(unseen).sum():,} test flows the model has never seen an example of\n")

cols = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "count", "serror_rate", "same_srv_rate"]
print("4) A normal web flow vs a SYN-flood (neptune) flow:")
example = pd.concat([
    train[(train.label == "normal") & (train.service == "http")].head(1),
    train[train.label == "neptune"].head(1),
])[cols + ["label"]]
print(example.to_string(index=False), "\n")

print("5) Average of 3 features by category (look for what separates them):")
print(train.groupby("category")[["src_bytes", "count", "serror_rate"]].mean().reindex(CATEGORIES).round(2).to_string())
