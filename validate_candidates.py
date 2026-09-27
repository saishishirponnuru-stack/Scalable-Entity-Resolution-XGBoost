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
print("Checking candidate_pairs_final.tsv...")

rows = 0
sources = set()
empty = 0
duplicate_sources = 0
invalid = 0
duplicate_candidates = 0
total_candidates = 0

for chunk in pd.read_csv(
    "output/candidate_pairs_final.tsv",
    sep="\t",
    dtype=str,
    chunksize=50000
):
    for _, r in chunk.iterrows():

        sid = r["source1_entity_id"]

        if sid in sources:
            duplicate_sources += 1
        sources.add(sid)

        value = str(r["candidate_entity_ids"])

        if not value or value == "nan":
            empty += 1
            continue

        ids = value.split(",")

        rows += 1
        total_candidates += len(ids)

        duplicate_candidates += len(ids) - len(set(ids))
        invalid += sum(i not in valid for i in ids)

print("Unique Source1 entities:", len(sources))
print("Empty candidate lists:", empty)
print("Duplicate Source1 rows:", duplicate_sources)
print("Total candidate IDs:", total_candidates)
print("Invalid candidate IDs:", invalid)
print("Duplicate candidate IDs:", duplicate_candidates)