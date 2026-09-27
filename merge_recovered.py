import pandas as pd

MAIN = "output/candidate_pairs.tsv"
RECOVERED = "output/recovered_candidates.tsv"
OUT = "output/candidate_pairs_final.tsv"

main = pd.read_csv(MAIN, sep="\t", dtype=str).fillna("")
rec = pd.read_csv(RECOVERED, sep="\t", dtype=str).fillna("")

lookup = dict(
    zip(
        rec["source1_entity_id"],
        rec["candidate_entity_ids"]
    )
)

mask = main["source1_entity_id"].isin(lookup)

main.loc[mask, "candidate_entity_ids"] = (
    main.loc[mask, "source1_entity_id"].map(lookup)
)

main.to_csv(OUT, sep="\t", index=False)

print("Original rows:", len(main))
print("Recovered rows merged:", mask.sum())
print("Empty candidate lists:", (main["candidate_entity_ids"] == "").sum())
print("Unique Source1:", main["source1_entity_id"].nunique())
print("Duplicate Source1:", main["source1_entity_id"].duplicated().sum())
print("Output:", OUT)
