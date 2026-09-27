import pandas as pd

print("Loading target IDs...")

s2 = pd.read_csv(
    "clean_test_data/clean_source2.tsv",
    sep="\t",
    usecols=["entity_id"],
    dtype=str
)

s3 = pd.read_csv(
    "clean_test_data/clean_source3.tsv",
    sep="\t",
    usecols=["entity_id"],
    dtype=str
)

valid = set(s2["entity_id"]) | set(s3["entity_id"])

print("Target pool:", len(valid))

total = 0
bad = 0
dup = 0

print("Checking matching_results_093.tsv...")

for chunk in pd.read_csv(
    "output/matching_results_093.tsv",
    sep="\t",
    dtype=str,
    chunksize=50000
):
    for value in chunk["matched_entity_ids"].fillna(""):
        if not value:
            continue

        ids = value.split(",")
        total += len(ids)

        dup += len(ids) - len(set(ids))
        bad += sum(i not in valid for i in ids)

print("Predicted links checked:", total)
print("Invalid target IDs:", bad)
print("Duplicate IDs:", dup)