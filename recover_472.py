import pandas as pd
import re
from rapidfuzz import process, fuzz

SOURCE = "clean_test_data/clean_source1.tsv"
TARGETS = [
    "clean_test_data/clean_source2.tsv",
    "clean_test_data/clean_source3.tsv"
]
CAND = "output/candidate_pairs.tsv"
OUT = "output/recovered_candidates.tsv"

print("[1/4] Loading empty-candidate Source1 entities...")

cand = pd.read_csv(CAND, sep="\t", dtype=str).fillna("")
empty_ids = set(cand.loc[cand["candidate_entity_ids"] == "", "source1_entity_id"])

print("Empty candidates:", len(empty_ids))

s1 = pd.read_csv(
    SOURCE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "clean_name",
        "clean_address",
        "clean_country"
    ]
).fillna("")

missing = s1[s1["entity_id"].isin(empty_ids)].copy()

print("Entities to recover:", len(missing))

print("[2/4] Loading TEST target names...")

parts = []

for path in TARGETS:
    print("Loading:", path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "clean_name",
            "clean_address",
            "clean_country"
        ]
    ).fillna("")
    parts.append(df)

targets = pd.concat(parts, ignore_index=True)

print("Target entities:", len(targets))

print("[3/4] Building compact name index...")

def compact(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

targets["compact"] = targets["clean_name"].map(compact)

name_choices = {}
for i, value in enumerate(targets["clean_name"]):
    if value:
        name_choices.setdefault(value, i)

compact_choices = {}
for i, value in enumerate(targets["compact"]):
    if value:
        compact_choices.setdefault(value, i)

print("Unique names:", len(name_choices))
print("Unique compact names:", len(compact_choices))

print("[4/4] Recovering candidates...")

rows = []

for n, (_, src) in enumerate(missing.iterrows(), 1):

    name = src["clean_name"]
    addr = src["clean_address"]
    country = src["clean_country"]

    selected = {}

    # A. Exact clean-name match
    if name and name in name_choices:
        i = name_choices[name]
        selected[targets.iloc[i]["entity_id"]] = 100

    # B. Compact-name match
    c = compact(name)
    if c and c in compact_choices:
        i = compact_choices[c]
        selected[targets.iloc[i]["entity_id"]] = 99

    # C. Fuzzy name retrieval
    if name:
        results = process.extract(
            name,
            name_choices.keys(),
            scorer=fuzz.ratio,
            limit=15,
            score_cutoff=55
        )

        for match_name, score, _ in results:
            i = name_choices[match_name]
            eid = targets.iloc[i]["entity_id"]

            # Give address/country a small confirmation boost
            address_score = fuzz.ratio(
                addr,
                targets.iloc[i]["clean_address"]
            ) if addr and targets.iloc[i]["clean_address"] else 0

            country_bonus = (
                5 if country and country == targets.iloc[i]["clean_country"]
                else 0
            )

            final_score = 0.70 * score + 0.25 * address_score + country_bonus

            if final_score >= 45:
                selected[eid] = max(
                    selected.get(eid, 0),
                    final_score
                )

    # Keep at most 15 strongest candidates
    ranked = sorted(
        selected.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    ids = ",".join(x[0] for x in ranked)

    rows.append({
        "source1_entity_id": src["entity_id"],
        "candidate_entity_ids": ids
    })

    if n % 50 == 0 or n == len(missing):
        print(f"Recovered {n}/{len(missing)}")

out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    sep="\t",
    index=False
)

print()
print("RECOVERY COMPLETE")
print("Rows:", len(out))
print("Recovered:", (out["candidate_entity_ids"] != "").sum())
print("Still empty:", (out["candidate_entity_ids"] == "").sum())
print("Output:", OUT)
