import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from xgboost import XGBClassifier

CANDIDATES = "output/candidate_pairs_final.tsv"
MODEL = "output/xgb_best_v2.json"
OUTPUT = "output/matching_results_093.tsv"

THRESHOLD = 0.93
CHUNK_SIZE = 5000

FEATURES = [
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

print("[1/5] Loading TEST data...")

s1 = pd.read_csv(
    "clean_test_data/clean_source1.tsv",
    sep="\t",
    dtype=str
).fillna("")

s2 = pd.read_csv(
    "clean_test_data/clean_source2.tsv",
    sep="\t",
    dtype=str
).fillna("")

s3 = pd.read_csv(
    "clean_test_data/clean_source3.tsv",
    sep="\t",
    dtype=str
).fillna("")

target = pd.concat([s2, s3], ignore_index=True)
target = target.set_index("entity_id")

source = s1.set_index("entity_id")

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))
print("Target pool:", len(target))

print()
print("[2/5] Loading XGBoost model...")

model = XGBClassifier()
model.load_model(MODEL)

print("Model:", MODEL)
print("Threshold:", THRESHOLD)

print()
print("[3/5] Scoring candidate pairs...")

results = []

reader = pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
)

processed = 0
scored = 0
matched = 0

for chunk_no, df in enumerate(reader, 1):

    df = df.fillna("")

    rows = []

    for _, r in df.iterrows():

        sid = r["source1_entity_id"]
        candidate_string = r["candidate_entity_ids"]

        if not candidate_string:
            continue

        if sid not in source.index:
            continue

        s = source.loc[sid]

        for tid in candidate_string.split(","):

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

                "name_ratio": fuzz.ratio(n1, n2),
                "name_token_sort": fuzz.token_sort_ratio(n1, n2),
                "name_token_set": fuzz.token_set_ratio(n1, n2),
                "name_partial": fuzz.partial_ratio(n1, n2),

                "addr_ratio": fuzz.ratio(a1, a2),
                "addr_token_set": fuzz.token_set_ratio(a1, a2),
                "addr_partial": fuzz.partial_ratio(a1, a2),

                "country_match": int(
                    n1 != "" and
                    s["clean_country"] != "" and
                    s["clean_country"] == t["clean_country"]
                ),

                "missing_addr": int(
                    a1 == "" or a2 == ""
                ),

                "name_len_diff":
                    abs(len(n1) - len(n2)) /
                    max(max(len(n1), len(n2)), 1),

                "addr_len_diff":
                    abs(len(a1) - len(a2)) /
                    max(max(len(a1), len(a2)), 1)
            })

    if rows:

        feat = pd.DataFrame(rows)

        probabilities = model.predict_proba(
            feat[FEATURES]
        )[:, 1]

        feat["probability"] = probabilities

        keep = feat[feat["probability"] >= THRESHOLD]

        if len(keep):
            results.append(
                keep[
                    ["source1_entity_id", "candidate_id"]
                ]
            )

            matched += len(keep)

        scored += len(feat)

    processed += len(df)

    if chunk_no % 25 == 0:
        print(
            f"Chunks: {chunk_no} | "
            f"Source1 processed: {processed:,} | "
            f"Scored: {scored:,} | "
            f"Matches: {matched:,}"
        )

print()
print("[4/5] Building final matching output...")

if results:
    matches = pd.concat(results, ignore_index=True)

    grouped = (
        matches
        .groupby("source1_entity_id")["candidate_id"]
        .apply(",".join)
        .reset_index()
    )

else:
    grouped = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_id"
        ]
    )

grouped = grouped.rename(
    columns={
        "candidate_id": "matched_entity_ids"
    }
)

final = s1[
    ["entity_id"]
].rename(
    columns={
        "entity_id": "source1_entity_id"
    }
)

final = final.merge(
    grouped,
    on="source1_entity_id",
    how="left"
)

final["matched_entity_ids"] = (
    final["matched_entity_ids"]
    .fillna("")
)

final[
    ["source1_entity_id", "matched_entity_ids"]
].to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print()
print("[5/5] COMPLETE")
print("Source1 entities:", len(final))
print("Matched Source1:", (final["matched_entity_ids"] != "").sum())
print("Empty matches:", (final["matched_entity_ids"] == "").sum())
print("Total predicted links:", matched)
print("Output:", OUTPUT)