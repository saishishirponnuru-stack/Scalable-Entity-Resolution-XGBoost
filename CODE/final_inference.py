import os
import gc
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio as token_sort,
    token_set_ratio as token_set,
    partial_ratio as partial
)

import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

CANDIDATE_FILE = "output/candidate_pairs.tsv"
MODEL_FILE = "output/xgb_best_v2.json"
OUTPUT_FILE = "output/matching_results.tsv"

SOURCE1_FILE = "clean_test_data/clean_source1.tsv"
SOURCE2_FILE = "clean_test_data/clean_source2.tsv"
SOURCE3_FILE = "clean_test_data/clean_source3.tsv"

THRESHOLD = 0.88

# Process source rows in chunks.
SOURCE_CHUNK = 5000

# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL TEST INFERENCE")
print("=" * 70)

print("\n[1/5] Loading TEST data...")

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "clean_name",
        "clean_address",
        "clean_country"
    ]
)

s2 = pd.read_csv(
    SOURCE2_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "clean_name",
        "clean_address",
        "clean_country"
    ]
)

s3 = pd.read_csv(
    SOURCE3_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "clean_name",
        "clean_address",
        "clean_country"
    ]
)

target = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(f"Source 1:    {len(s1):,}")
print(f"Source 2:    {len(s2):,}")
print(f"Source 3:    {len(s3):,}")
print(f"Target pool: {len(target):,}")


# ============================================================
# PREPARE LOOKUPS
# ============================================================

print("\n[2/5] Preparing lookups...")

s1 = s1.set_index("entity_id")

target = target.set_index("entity_id")

# Make sure IDs are strings.
s1.index = s1.index.astype(str)
target.index = target.index.astype(str)

print("Lookup preparation complete.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[3/5] Loading XGBoost model...")

model = xgb.XGBClassifier()

model.load_model(MODEL_FILE)

print(f"Model: {MODEL_FILE}")
print(f"Threshold: {THRESHOLD}")


# ============================================================
# OUTPUT INITIALIZATION
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_write = True

total_sources = 0
total_candidates = 0
total_matches = 0


# ============================================================
# FEATURE FUNCTION
# ============================================================

def build_features(df):
    """
    Build the exact 11 features used during training.
    """

    text_cols = [
        "s1_name",
        "s1_address",
        "s1_country",
        "cand_name",
        "cand_address",
        "cand_country"
    ]

    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

    df["country_match"] = (
        df["s1_country"] ==
        df["cand_country"]
    ).astype(np.int8)

    df["missing_addr"] = (
        (df["s1_address"] == "") |
        (df["cand_address"] == "")
    ).astype(np.int8)

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    names1 = df["s1_name"].tolist()
    names2 = df["cand_name"].tolist()

    df["name_ratio"] = [
        ratio(a, b)
        for a, b in zip(names1, names2)
    ]

    df["name_token_sort"] = [
        token_sort(a, b)
        for a, b in zip(names1, names2)
    ]

    df["name_token_set"] = [
        token_set(a, b)
        for a, b in zip(names1, names2)
    ]

    df["name_partial"] = [
        partial(a, b)
        for a, b in zip(names1, names2)
    ]

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    addr1 = df["s1_address"].tolist()
    addr2 = df["cand_address"].tolist()

    df["addr_ratio"] = [
        ratio(a, b)
        for a, b in zip(addr1, addr2)
    ]

    df["addr_token_set"] = [
        token_set(a, b)
        for a, b in zip(addr1, addr2)
    ]

    df["addr_partial"] = [
        partial(a, b)
        for a, b in zip(addr1, addr2)
    ]

    # --------------------------------------------------------
    # LENGTH FEATURES
    # --------------------------------------------------------

    n1 = df["s1_name"].str.len()
    n2 = df["cand_name"].str.len()

    df["name_len_diff"] = (
        (n1 - n2).abs() /
        np.maximum(
            np.maximum(n1, n2),
            1
        )
    )

    a1 = df["s1_address"].str.len()
    a2 = df["cand_address"].str.len()

    df["addr_len_diff"] = (
        (a1 - a2).abs() /
        np.maximum(
            np.maximum(a1, a2),
            1
        )
    )

    return df


# ============================================================
# PROCESS CANDIDATES
# ============================================================

print("\n[4/5] Scoring TEST candidates...")

candidate_reader = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    chunksize=SOURCE_CHUNK
)

for chunk_no, chunk in enumerate(candidate_reader, start=1):

    # --------------------------------------------------------
    # Parse candidate lists
    # --------------------------------------------------------

    rows = []

    for _, row in chunk.iterrows():

        source_id = str(row["source1_entity_id"])

        candidates = str(
            row["candidate_entity_ids"]
        )

        if candidates == "" or candidates == "nan":
            continue

        candidate_ids = [
            x.strip()
            for x in candidates.split(",")
            if x.strip()
        ]

        for candidate_id in candidate_ids:

            rows.append(
                (
                    source_id,
                    candidate_id
                )
            )

    if not rows:
        continue

    pair_df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_id"
        ]
    )

    # --------------------------------------------------------
    # SOURCE LOOKUP
    # --------------------------------------------------------

    source_features = s1.reindex(
        pair_df["source1_entity_id"].values
    ).reset_index(drop=True)

    source_features = source_features.rename(
        columns={
            "clean_name": "s1_name",
            "clean_address": "s1_address",
            "clean_country": "s1_country"
        }
    )

    # --------------------------------------------------------
    # TARGET LOOKUP
    # --------------------------------------------------------

    target_features = target.reindex(
        pair_df["candidate_id"].values
    ).reset_index(drop=True)

    target_features = target_features.rename(
        columns={
            "clean_name": "cand_name",
            "clean_address": "cand_address",
            "clean_country": "cand_country"
        }
    )

    df = pd.concat(
        [
            pair_df.reset_index(drop=True),
            source_features,
            target_features
        ],
        axis=1
    )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    df = build_features(df)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    X = df[FEATURE_COLS].astype(np.float32)

    probabilities = model.predict_proba(X)[:, 1]

    df["probability"] = probabilities

    # --------------------------------------------------------
    # KEEP MATCHES
    # --------------------------------------------------------

    matches = df[
        df["probability"] >= THRESHOLD
    ][
        [
            "source1_entity_id",
            "candidate_id",
            "probability"
        ]
    ].copy()

    # --------------------------------------------------------
    # WRITE MATCHES
    # --------------------------------------------------------

    if len(matches) > 0:

        # Keep candidates sorted by model confidence.
        matches = matches.sort_values(
            [
                "source1_entity_id",
                "probability"
            ],
            ascending=[
                True,
                False
            ]
        )

        matches["candidate_id"] = (
            matches["candidate_id"].astype(str)
        )

        # Group matched IDs by source entity.
        grouped = (
            matches
            .groupby(
                "source1_entity_id",
                sort=False
            )["candidate_id"]
            .apply(",".join)
            .reset_index()
        )

        grouped.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        total_matches += len(matches)

    total_sources += len(chunk)
    total_candidates += len(pair_df)

    print(
        f"Chunk {chunk_no:,} | "
        f"Sources: {total_sources:,} | "
        f"Candidates: {total_candidates:,} | "
        f"Matches: {total_matches:,}"
    )

    del pair_df
    del source_features
    del target_features
    del df
    del X
    del matches

    gc.collect()


# ============================================================
# FINISH
# ============================================================

print("\n[5/5] FINAL INFERENCE COMPLETE")

print("=" * 70)
print("DONE")
print("=" * 70)

print(f"Test Source 1 entities processed: {total_sources:,}")
print(f"Candidate pairs scored:           {total_candidates:,}")
print(f"Matched candidate pairs:          {total_matches:,}")
print(f"Output:                            {OUTPUT_FILE}")
print("=" * 70)