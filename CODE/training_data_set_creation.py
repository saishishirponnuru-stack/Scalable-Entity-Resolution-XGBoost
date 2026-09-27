import pandas as pd
import numpy as np
import os
from rapidfuzz import fuzz

print("=" * 70)
print("ROBUST STREAMING TRAINING FEATURE GENERATOR")
print("=" * 70)

CANDIDATE_FILE = "output/candidate_pairs.tsv"
OUTPUT_FILE = "output/ml_training_features_v4.tsv"

S1_FILE = "clean_train_data/clean_source1.tsv"
S2_FILE = "clean_train_data/clean_source2.tsv"
S3_FILE = "clean_train_data/clean_source3.tsv"
GT_FILE = "train/train_ground_truth.tsv"

CHUNK_SIZE = 25000
NEGATIVES_PER_SOURCE = 8


# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def ratio(a, b):
    return fuzz.ratio(a, b) / 100.0


def token_sort(a, b):
    return fuzz.token_sort_ratio(a, b) / 100.0


def token_set(a, b):
    return fuzz.token_set_ratio(a, b) / 100.0


def partial(a, b):
    return fuzz.partial_ratio(a, b) / 100.0


# ============================================================
# 1. LOAD SOURCE 1
# ============================================================

print("\n[1/6] Loading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "clean_name",
        "clean_address",
        "clean_country"
    ]
)

# IMPORTANT:
# Make entity_id unique so lookup always returns ONE row.
s1 = s1.drop_duplicates(
    subset="entity_id",
    keep="first"
)

s1 = s1.set_index("entity_id")

print(f"Unique Source 1: {len(s1):,}")


# ============================================================
# 2. LOAD TARGET POOL
# ============================================================

print("\n[2/6] Loading target pool...")

cols = [
    "entity_id",
    "clean_name",
    "clean_address",
    "clean_country"
]

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=cols
)

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=cols
)

target = pd.concat(
    [s2, s3],
    ignore_index=True
)

# IMPORTANT:
# Entity IDs should identify one target record.
target = target.drop_duplicates(
    subset="entity_id",
    keep="first"
)

target = target.set_index("entity_id")

print(f"Unique target entities: {len(target):,}")


# ============================================================
# 3. LOAD GROUND TRUTH
# ============================================================

print("\n[3/6] Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

# Build one row per actual match.
gt_rows = []

for _, row in gt.iterrows():

    sid = row["source1_entity_id"]
    value = row["matched_entity_ids"]

    if pd.isna(value):
        continue

    for cid in str(value).split(","):

        cid = cid.strip()

        if cid:
            gt_rows.append(
                {
                    "source1_entity_id": sid,
                    "candidate_id": cid,
                    "is_match": 1
                }
            )

gt_pairs = pd.DataFrame(
    gt_rows
)

print(
    f"Ground-truth match pairs: "
    f"{len(gt_pairs):,}"
)

# Use a MultiIndex for extremely fast membership testing.
gt_index = pd.MultiIndex.from_frame(
    gt_pairs[
        [
            "source1_entity_id",
            "candidate_id"
        ]
    ]
)


# ============================================================
# 4. PREPARE OUTPUT
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_write = True

total_candidates = 0
total_features = 0


# ============================================================
# 5. STREAM CANDIDATES
# ============================================================

print("\n[4/6] Streaming candidate pairs...")
print(
    f"Chunk size: {CHUNK_SIZE:,}"
)

reader = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
)


for chunk_no, candidates in enumerate(
    reader,
    start=1
):

    print(
        f"\nChunk {chunk_no}: "
        f"{len(candidates):,} source rows"
    )

    candidates = candidates.dropna(
        subset=["candidate_entity_ids"]
    )

    # --------------------------------------------------------
    # Expand ONLY this chunk.
    # --------------------------------------------------------

    candidates["candidate_id"] = (
        candidates["candidate_entity_ids"]
        .astype(str)
        .str.split(",")
    )

    df = candidates[
        [
            "source1_entity_id",
            "candidate_id"
        ]
    ].explode(
        "candidate_id",
        ignore_index=True
    )

    df["candidate_id"] = (
        df["candidate_id"]
        .astype(str)
        .str.strip()
    )

    total_candidates += len(df)

    # --------------------------------------------------------
    # VECTORISED LABELING
    # --------------------------------------------------------

    pair_index = pd.MultiIndex.from_frame(
        df[
            [
                "source1_entity_id",
                "candidate_id"
            ]
        ]
    )

    df["is_match"] = (
        pair_index.isin(gt_index)
    ).astype(np.int8)

    # --------------------------------------------------------
    # SOURCE LOOKUP
    # --------------------------------------------------------

    source_features = s1[
        [
            "clean_name",
            "clean_address",
            "clean_country"
        ]
    ].rename(
        columns={
            "clean_name": "s1_name",
            "clean_address": "s1_address",
            "clean_country": "s1_country"
        }
    )

    df = df.join(
        source_features,
        on="source1_entity_id"
    )

    # --------------------------------------------------------
    # TARGET LOOKUP
    # --------------------------------------------------------

    target_features = target[
        [
            "clean_name",
            "clean_address",
            "clean_country"
        ]
    ].rename(
        columns={
            "clean_name": "cand_name",
            "clean_address": "cand_address",
            "clean_country": "cand_country"
        }
    )

    df = df.join(
        target_features,
        on="candidate_id"
    )

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    text_cols = [
        "s1_name",
        "s1_address",
        "s1_country",
        "cand_name",
        "cand_address",
        "cand_country"
    ]

    for col in text_cols:
        df[col] = df[col].fillna("")

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

    # --------------------------------------------------------
    # HARD NEGATIVE SAMPLING
    # --------------------------------------------------------

    feature_cols = [
        "source1_entity_id",
        "candidate_id",
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
        "addr_len_diff",
        "is_match"
    ]

    df = df[feature_cols]

    positives = df[
        df["is_match"] == 1
    ].copy()

    negatives = df[
        df["is_match"] == 0
    ].copy()

    # Hard-negative score.
    negatives["hard_score"] = (
        0.45 * negatives["name_token_set"] +
        0.25 * negatives["name_ratio"] +
        0.15 * negatives["name_token_sort"] +
        0.15 * negatives["addr_token_set"]
    )

    negatives = (
        negatives
        .sort_values(
            [
                "source1_entity_id",
                "hard_score"
            ],
            ascending=[
                True,
                False
            ]
        )
        .groupby(
            "source1_entity_id",
            sort=False
        )
        .head(NEGATIVES_PER_SOURCE)
    )

    negatives = negatives.drop(
        columns=["hard_score"]
    )

    training = pd.concat(
        [
            positives,
            negatives
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    training.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write
    )

    first_write = False

    total_features += len(training)

    print(
        f"Candidate pairs: {len(df):,}"
    )

    print(
        f"Positives: {len(positives):,}"
    )

    print(
        f"Hard negatives: {len(negatives):,}"
    )

    print(
        f"Written training rows: "
        f"{len(training):,}"
    )

    print(
        f"Total training rows so far: "
        f"{total_features:,}"
    )


# ============================================================
# 6. DONE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FEATURE GENERATION COMPLETE")
print("=" * 70)

print(
    f"Candidate pairs processed: "
    f"{total_candidates:,}"
)

print(
    f"Training rows written: "
    f"{total_features:,}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)

print("=" * 70)