import pandas as pd

BASE = "output/matching_results_final.tsv"
REC = "output/recovered_matches.tsv"
OUT = "output/matching_results_final_v2.tsv"

base = pd.read_csv(BASE, sep="\t", dtype=str).fillna("")
rec = pd.read_csv(REC, sep="\t", dtype=str).fillna("")

lookup = dict(
    zip(
        rec["source1_entity_id"],
        rec["candidate_id"]
    )
)

mask = base["source1_entity_id"].isin(lookup)

base.loc[mask, "matched_entity_ids"] = (
    base.loc[mask, "source1_entity_id"].map(lookup)
)

base.to_csv(
    OUT,
    sep="\t",
    index=False
)

print("Total Source1 rows:", len(base))
print("Recovered rows merged:", mask.sum())
print("Empty match rows:", (base["matched_entity_ids"] == "").sum())
print("Duplicate Source1:", base["source1_entity_id"].duplicated().sum())
print("Output:", OUT)
