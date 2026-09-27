import pandas as pd

r = pd.read_csv(
    "output/matching_results_final.tsv",
    sep="\t",
    dtype=str
).fillna("")

s2 = set(pd.read_csv(
    "clean_test_data/clean_source2.tsv",
    sep="\t",
    usecols=["entity_id"],
    dtype=str
)["entity_id"])

s3 = set(pd.read_csv(
    "clean_test_data/clean_source3.tsv",
    sep="\t",
    usecols=["entity_id"],
    dtype=str
)["entity_id"])

valid = s2 | s3

bad = 0
dup = 0
total = 0
rows_with_matches = 0
max_matches = 0

for x in r["matched_entity_ids"]:
    if not x:
        continue

    ids = x.split(",")

    total += len(ids)
    rows_with_matches += 1
    max_matches = max(max_matches, len(ids))

    if len(ids) != len(set(ids)):
        dup += 1

    bad += sum(i not in valid for i in ids)

print("Valid target IDs:", len(valid))
print("Rows:", len(r))
print("Empty match rows:", (r["matched_entity_ids"] == "").sum())
print("Rows with matches:", rows_with_matches)
print("Total predicted links:", total)
print("Maximum matches for one Source1:", max_matches)
print("Duplicate-ID rows:", dup)
print("Invalid target IDs:", bad)
