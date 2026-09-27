import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from xgboost import XGBClassifier

MODEL = "output/xgb_best_v2.json"
CAND = "output/recovered_candidates.tsv"

print("[1/4] Loading recovered candidates...")

rec = pd.read_csv(
    CAND,
    sep="\t",
    dtype=str
).fillna("")

s1 = pd.read_csv(
    "clean_test_data/clean_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=["entity_id","clean_name","clean_address","clean_country"]
).fillna("")

parts = []

for f in [
    "clean_test_data/clean_source2.tsv",
    "clean_test_data/clean_source3.tsv"
]:
    parts.append(pd.read_csv(
        f,
        sep="\t",
        dtype=str,
        usecols=["entity_id","clean_name","clean_address","clean_country"]
    ).fillna(""))

target = pd.concat(parts, ignore_index=True)
target = target.set_index("entity_id")

src = s1.set_index("entity_id")

rows = []

for _, r in rec.iterrows():

    sid = r["source1_entity_id"]

    if not r["candidate_entity_ids"]:
        continue

    s = src.loc[sid]

    for tid in r["candidate_entity_ids"].split(","):

        if tid not in target.index:
            continue

        t = target.loc[tid]

        n1 = s["clean_name"]
        n2 = t["clean_name"]

        a1 = s["clean_address"]
        a2 = t["clean_address"]

        rows.append({
            "source1_entity_id": sid,
            "candidate_id": tid,

            "name_ratio": fuzz.ratio(n1,n2),
            "name_token_sort": fuzz.token_sort_ratio(n1,n2),
            "name_token_set": fuzz.token_set_ratio(n1,n2),
            "name_partial": fuzz.partial_ratio(n1,n2),

            "addr_ratio": fuzz.ratio(a1,a2),
            "addr_token_set": fuzz.token_set_ratio(a1,a2),
            "addr_partial": fuzz.partial_ratio(a1,a2),

            "country_match": int(
                s["clean_country"] != "" and
                s["clean_country"] == t["clean_country"]
            ),

            "missing_addr": int(
                a1 == "" or a2 == ""
            ),

            "name_len_diff": abs(len(n1)-len(n2)) /
                max(max(len(n1),len(n2)),1),

            "addr_len_diff": abs(len(a1)-len(a2)) /
                max(max(len(a1),len(a2)),1)
        })

df = pd.DataFrame(rows)

print("Recovered candidate pairs:", len(df))

if len(df) == 0:
    print("No candidates to score.")
    raise SystemExit

print("[2/4] Loading XGBoost model...")

model = XGBClassifier()
model.load_model(MODEL)

features = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "addr_ratio",
    "addr_token_set",
    "addr_partial",
    "country_match",
    "missing_addr",
    "name_len_diff",
    "addr_len_diff"
]

print("[3/4] Predicting...")

df["probability"] = model.predict_proba(
    df[features]
)[:,1]

matches = df[df["probability"] >= 0.88].copy()

print("Recovered candidates:", len(df))
print("Recovered matches:", len(matches))

print("[4/4] Writing recovered matches...")

out = (
    matches
    .groupby("source1_entity_id")["candidate_id"]
    .apply(",".join)
    .reset_index()
)

out.to_csv(
    "output/recovered_matches.tsv",
    sep="\t",
    index=False
)

print()
print("RECOVERED MATCH INFERENCE COMPLETE")
print("Source1 with recovered matches:", len(out))
print("Output: output/recovered_matches.tsv")
