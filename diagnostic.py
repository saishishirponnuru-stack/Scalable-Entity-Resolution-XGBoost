import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from rapidfuzz import fuzz

MODEL = "output/xgb_best_v2.json"

print("Loading model...")
model = XGBClassifier()
model.load_model(MODEL)

print("Loading TEST Source1...")
s1 = pd.read_csv(
    "clean_test_data/clean_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=20000
).fillna("")

print("Loading targets...")
t2 = pd.read_csv(
    "clean_test_data/clean_source2.tsv",
    sep="\t",
    dtype=str
).fillna("")

t3 = pd.read_csv(
    "clean_test_data/clean_source3.tsv",
    sep="\t",
    dtype=str
).fillna("")

targets = pd.concat([t2, t3], ignore_index=True)
targets = targets.set_index("entity_id")

print("Loading candidate sample...")
c = pd.read_csv(
    "output/candidate_pairs.tsv",
    sep="\t",
    dtype=str,
    nrows=20000
).fillna("")

rows = []

for _, r in c.iterrows():
    if not r["candidate_entity_ids"]:
        continue

    for tid in r["candidate_entity_ids"].split(","):
        rows.append((r["source1_entity_id"], tid))

d = pd.DataFrame(
    rows,
    columns=["sid", "tid"]
)

d = d[d["tid"].isin(targets.index)].copy()

print("Candidate pairs sampled:", len(d))

src = s1.set_index("entity_id")

a = src.loc[d["sid"]]
b = targets.loc[d["tid"]]

features = pd.DataFrame({
    "name_ratio": [
        fuzz.ratio(x, y)
        for x, y in zip(a.clean_name, b.clean_name)
    ],

    "name_token_sort": [
        fuzz.token_sort_ratio(x, y)
        for x, y in zip(a.clean_name, b.clean_name)
    ],

    "name_token_set": [
        fuzz.token_set_ratio(x, y)
        for x, y in zip(a.clean_name, b.clean_name)
    ],

    "name_partial": [
        fuzz.partial_ratio(x, y)
        for x, y in zip(a.clean_name, b.clean_name)
    ],

    "addr_ratio": [
        fuzz.ratio(x, y)
        for x, y in zip(a.clean_address, b.clean_address)
    ],

    "addr_token_set": [
        fuzz.token_set_ratio(x, y)
        for x, y in zip(a.clean_address, b.clean_address)
    ],

    "addr_partial": [
        fuzz.partial_ratio(x, y)
        for x, y in zip(a.clean_address, b.clean_address)
    ],

    "country_match": [
        int(x != "" and x == y)
        for x, y in zip(a.clean_country, b.clean_country)
    ],

    "missing_addr": [
        int(x == "" or y == "")
        for x, y in zip(a.clean_address, b.clean_address)
    ],

    "name_len_diff": [
        abs(len(x) - len(y)) /
        max(max(len(x), len(y)), 1)
        for x, y in zip(a.clean_name, b.clean_name)
    ],

    "addr_len_diff": [
        abs(len(x) - len(y)) /
        max(max(len(x), len(y)), 1)
        for x, y in zip(a.clean_address, b.clean_address)
    ]
})

print("Predicting...")
p = model.predict_proba(features)[:, 1]

print()
print("========== MODEL DIAGNOSTIC ==========")
print("Candidate pairs sampled:", len(p))
print(">= 0.99:", int((p >= 0.99).sum()))
print(">= 0.95:", int((p >= 0.95).sum()))
print(">= 0.90:", int((p >= 0.90).sum()))
print(">= 0.88:", int((p >= 0.88).sum()))
print(">= 0.80:", int((p >= 0.80).sum()))
print()
print("Median probability:", round(float(np.median(p)), 4))
print("Mean probability:", round(float(np.mean(p)), 4))
print("Minimum probability:", round(float(np.min(p)), 4))
print("Maximum probability:", round(float(np.max(p)), 4))
print("======================================")