import pandas as pd
import os

print("=" * 70)
print("STAGE 1 — CANDIDATE RECALL AUDIT")
print("=" * 70)

CANDIDATE_FILE = "output/candidate_pairs.tsv"
GROUND_TRUTH_FILE = "train/train_ground_truth.tsv"

# ------------------------------------------------------------
# 1. Load first 10,000 source entities from candidate file
# ------------------------------------------------------------

print("\n[1/4] Loading 10,000 candidate rows...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    nrows=10000
)

print(f"Candidate rows loaded: {len(candidates):,}")
print(f"Columns: {list(candidates.columns)}")

# Automatically identify the important columns
source_col = "source1_entity_id"

candidate_col = None
for col in candidates.columns:
    if col != source_col:
        candidate_col = col
        break

if source_col not in candidates.columns:
    raise ValueError(
        f"Could not find '{source_col}' in candidate file."
    )

if candidate_col is None:
    raise ValueError("Could not identify candidate entity column.")

print(f"Source column: {source_col}")
print(f"Candidate column: {candidate_col}")

# ------------------------------------------------------------
# 2. Load ground truth
# ------------------------------------------------------------

print("\n[2/4] Loading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "matched_entity_ids"]
)

gt = gt[
    gt["source1_entity_id"].isin(
        candidates[source_col].astype(str)
    )
].copy()

print(f"Ground-truth rows for pilot: {len(gt):,}")

# ------------------------------------------------------------
# 3. Build ground-truth lookup
# ------------------------------------------------------------

print("\n[3/4] Comparing candidates with ground truth...")

gt_lookup = {}

for _, row in gt.iterrows():

    source_id = str(row["source1_entity_id"])
    value = row["matched_entity_ids"]

    if pd.isna(value):
        gt_ids = set()
    else:
        gt_ids = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_lookup[source_id] = gt_ids


# ------------------------------------------------------------
# 4. Calculate recall
# ------------------------------------------------------------

at_least_one_hit = 0
all_matches_hit = 0
evaluated = 0
total_candidates = 0

missed_examples = []

for _, row in candidates.iterrows():

    source_id = str(row[source_col])

    if source_id not in gt_lookup:
        continue

    true_ids = gt_lookup[source_id]

    if not true_ids:
        continue

    # Candidate IDs can be comma-separated
    candidate_ids = {
        x.strip()
        for x in str(row[candidate_col]).split(",")
        if x.strip()
    }

    total_candidates += len(candidate_ids)
    evaluated += 1

    intersection = candidate_ids & true_ids

    if intersection:
        at_least_one_hit += 1

    if true_ids.issubset(candidate_ids):
        all_matches_hit += 1

    if not intersection and len(missed_examples) < 10:
        missed_examples.append(
            (source_id, true_ids, candidate_ids)
        )


if evaluated == 0:
    raise RuntimeError(
        "No matching source IDs were found between "
        "candidate file and ground truth."
    )

recall_any = at_least_one_hit / evaluated
recall_all = all_matches_hit / evaluated
avg_candidates = total_candidates / evaluated

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Evaluated source entities : {evaluated:,}")
print(f"Average candidates/entity  : {avg_candidates:.2f}")
print()
print(
    f"At-least-one-match recall : "
    f"{recall_any:.4%}"
)
print(
    f"All-ground-truth recall   : "
    f"{recall_all:.4%}"
)

print()
print(f"Entities with >=1 hit      : {at_least_one_hit:,}")
print(f"Entities completely missed : {evaluated - at_least_one_hit:,}")

if missed_examples:

    print("\nExample missed entities:")
    print("-" * 70)

    for source_id, true_ids, candidate_ids in missed_examples:

        print(f"Source ID: {source_id}")
        print(f"True IDs : {', '.join(list(true_ids)[:5])}")
        print(f"Candidates: {', '.join(list(candidate_ids)[:5])}")
        print()

print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)