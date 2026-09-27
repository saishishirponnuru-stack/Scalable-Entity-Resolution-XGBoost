import pandas as pd
import re

print("=" * 70)
print("STAGE 1 — MISSED MATCH ANALYSIS")
print("=" * 70)

CANDIDATE_FILE = "output/candidate_pairs.tsv"
GROUND_TRUTH_FILE = "train/train_ground_truth.tsv"

S1_FILE = "clean_train_data/clean_source1.tsv"
S2_FILE = "clean_train_data/clean_source2.tsv"
S3_FILE = "clean_train_data/clean_source3.tsv"


# ------------------------------------------------------------
# Load first 10,000 generated candidates
# ------------------------------------------------------------

print("\n[1/5] Loading candidate data...")

cand = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    nrows=10000
)

cand["candidate_set"] = cand["candidate_entity_ids"].fillna("").apply(
    lambda x: {
        i.strip() for i in x.split(",") if i.strip()
    }
)

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

print("[2/5] Loading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "matched_entity_ids"]
)

gt = gt[
    gt["source1_entity_id"].isin(cand["source1_entity_id"])
].copy()

gt["true_set"] = gt["matched_entity_ids"].fillna("").apply(
    lambda x: {
        i.strip() for i in x.split(",") if i.strip()
    }
)

# ------------------------------------------------------------
# Find missed source entities
# ------------------------------------------------------------

print("[3/5] Finding missed matches...")

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["true_set"]
    )
)

missed = []

for _, row in cand.iterrows():

    sid = row["source1_entity_id"]

    if sid not in gt_map:
        continue

    true_ids = gt_map[sid]
    candidate_ids = row["candidate_set"]

    if true_ids and not (true_ids & candidate_ids):

        missed.append({
            "source1_entity_id": sid,
            "true_ids": list(true_ids),
            "candidate_ids": list(candidate_ids)
        })

print(f"Missed entities found: {len(missed):,}")

# We only inspect 20
missed = missed[:20]

true_target_ids = set()

for item in missed:
    true_target_ids.update(item["true_ids"])

candidate_target_ids = set()

for item in missed:
    candidate_target_ids.update(item["candidate_ids"])


# ------------------------------------------------------------
# Load relevant target records
# ------------------------------------------------------------

print("[4/5] Loading source and target records...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "clean_name",
        "clean_address",
        "clean_country"
    ]
)

needed_s1 = {
    x["source1_entity_id"] for x in missed
}

s1 = s1[
    s1["entity_id"].isin(needed_s1)
].copy()


def load_targets(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "clean_name",
            "clean_address",
            "clean_country"
        ]
    )

    needed = true_target_ids | candidate_target_ids

    return df[df["entity_id"].isin(needed)].copy()


print("Loading Source 2...")
s2 = load_targets(S2_FILE)

print("Loading Source 3...")
s3 = load_targets(S3_FILE)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

target_map = targets.set_index("entity_id").to_dict("index")
source_map = s1.set_index("entity_id").to_dict("index")


# ------------------------------------------------------------
# Print analysis
# ------------------------------------------------------------

print("\n[5/5] Printing missed-match examples...")
print()

for number, item in enumerate(missed, 1):

    sid = item["source1_entity_id"]

    src = source_map.get(sid)

    print("=" * 70)
    print(f"MISSED MATCH #{number}")
    print("=" * 70)

    print("\nSOURCE 1")
    print("ID      :", sid)
    print("Name    :", src.get("business_name"))
    print("Address :", src.get("business_address"))
    print("Country :", src.get("country"))
    print("Clean   :", src.get("clean_name"))

    print("\nTRUE TARGETS")

    for tid in item["true_ids"]:

        target = target_map.get(tid)

        if target:

            print("\nID      :", tid)
            print("Name    :", target.get("business_name"))
            print("Address :", target.get("business_address"))
            print("Country :", target.get("country"))
            print("Clean   :", target.get("clean_name"))

    print("\nGENERATED CANDIDATES")

    shown = 0

    for cid in item["candidate_ids"]:

        target = target_map.get(cid)

        if target:

            print(
                f"\nID      : {cid}"
            )
            print(
                f"Name    : {target.get('business_name')}"
            )
            print(
                f"Address : {target.get('business_address')}"
            )

            shown += 1

        if shown >= 5:
            break

print("\n" + "=" * 70)
print("MISSED MATCH ANALYSIS COMPLETE")
print("=" * 70)