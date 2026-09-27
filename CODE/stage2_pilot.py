import pandas as pd
import re
from rapidfuzz import fuzz, process

print("=" * 70)
print("STAGE 2 — HIGH RECALL RETRIEVAL PILOT")
print("=" * 70)

CANDIDATE_FILE = "output/candidate_pairs.tsv"
GROUND_TRUTH_FILE = "train/train_ground_truth.tsv"

S1_FILE = "clean_train_data/clean_source1.tsv"
S2_FILE = "clean_train_data/clean_source2.tsv"
S3_FILE = "clean_train_data/clean_source3.tsv"


# ============================================================
# HELPERS
# ============================================================

def normalize_compact(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def get_address_key(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Extract useful address numbers
    numbers = re.findall(r"\b\d+[a-z]?\b", text)

    # Keep reasonably distinctive numbers
    numbers = [
        x for x in numbers
        if len(x) >= 2
    ]

    return "|".join(numbers[:5])


def get_tokens(text):
    if pd.isna(text):
        return set()

    return {
        x for x in re.findall(
            r"[a-z0-9]+",
            str(text).lower()
        )
        if len(x) >= 3
    }


def char_ngrams(text, n=3):
    if not text:
        return set()

    text = normalize_compact(text)

    if len(text) < n:
        return {text}

    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


# ============================================================
# 1. LOAD PILOT SOURCE RECORDS
# ============================================================

print("\n[1/6] Loading pilot source records...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    nrows=10000
)

source_ids = candidates["source1_entity_id"].tolist()

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

s1 = s1[
    s1["entity_id"].isin(source_ids)
].copy()

print(f"Pilot source records: {len(s1):,}")


# ============================================================
# 2. LOAD TARGET DATA
# ============================================================

print("\n[2/6] Loading target pool...")

s2 = pd.read_csv(
    S2_FILE,
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

print(f"Source 2: {len(s2):,}")

s3 = pd.read_csv(
    S3_FILE,
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

print(f"Source 3: {len(s3):,}")

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(f"Target pool: {len(targets):,}")


# ============================================================
# 3. PREPARE FAST TARGET FEATURES
# ============================================================

print("\n[3/6] Preparing retrieval features...")

targets["compact_name"] = targets["clean_name"].map(
    normalize_compact
)

targets["address_key"] = targets["clean_address"].map(
    get_address_key
)

targets["name_tokens"] = targets["clean_name"].map(
    get_tokens
)

targets["char3"] = targets["clean_name"].map(
    char_ngrams
)

# Indexes
exact_index = {}

compact_index = {}

address_index = {}

token_index = {}

char_index = {}


def add_to_index(index, key, entity_id):

    if not key:
        return

    index.setdefault(key, []).append(entity_id)


print("Building indexes...")

for _, row in targets.iterrows():

    eid = row["entity_id"]

    add_to_index(
        exact_index,
        row["clean_name"],
        eid
    )

    add_to_index(
        compact_index,
        row["compact_name"],
        eid
    )

    add_to_index(
        address_index,
        row["address_key"],
        eid
    )

    for token in row["name_tokens"]:
        add_to_index(
            token_index,
            token,
            eid
        )

    # Only use first few character grams to keep index bounded
    for gram in list(row["char3"])[:20]:
        add_to_index(
            char_index,
            gram,
            eid
        )


print(f"Exact names   : {len(exact_index):,}")
print(f"Compact names : {len(compact_index):,}")
print(f"Addresses     : {len(address_index):,}")
print(f"Tokens        : {len(token_index):,}")
print(f"Char grams    : {len(char_index):,}")


# ============================================================
# 4. RETRIEVE CANDIDATES
# ============================================================

print("\n[4/6] Generating pilot candidates...")

target_rows = targets.set_index("entity_id")

results = {}

for counter, (_, row) in enumerate(s1.iterrows()):

    sid = row["entity_id"]

    found = set()

    name = row["clean_name"]
    compact = normalize_compact(name)
    address_key = get_address_key(row["clean_address"])
    tokens = get_tokens(name)
    grams = char_ngrams(name)

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    found.update(
        exact_index.get(name, [])
    )

    # --------------------------------------------------------
    # Compact name
    # --------------------------------------------------------

    found.update(
        compact_index.get(compact, [])
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    found.update(
        address_index.get(address_key, [])
    )

    # --------------------------------------------------------
    # Token blocking
    # --------------------------------------------------------

    for token in tokens:

        ids = token_index.get(token, [])

        # Avoid huge generic-token blocks
        if len(ids) <= 200:
            found.update(ids)

    # --------------------------------------------------------
    # Character blocking
    # --------------------------------------------------------

    for gram in grams:

        ids = char_index.get(gram, [])

        if len(ids) <= 200:
            found.update(ids)

    # --------------------------------------------------------
    # Country filter
    # --------------------------------------------------------

    country = str(row["clean_country"]).lower()

    if country and country != "nan":

        country_found = {
            eid
            for eid in found
            if str(
                target_rows.loc[eid, "clean_country"]
            ).lower() == country
        }

        if country_found:
            found = country_found

    results[sid] = found

    if counter % 1000 == 0:
        print(
            f"Processed {counter:,} / {len(s1):,}"
        )


# ============================================================
# 5. LOAD GROUND TRUTH
# ============================================================

print("\n[5/6] Evaluating against ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

gt = gt[
    gt["source1_entity_id"].isin(source_ids)
].copy()

gt_map = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value):
        true_ids = set()
    else:
        true_ids = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_map[row["source1_entity_id"]] = true_ids


# ============================================================
# 6. CALCULATE RECALL
# ============================================================

print("\n[6/6] Calculating recall...")

evaluated = 0
hit_any = 0
hit_all = 0
total_candidates = 0

for sid, found in results.items():

    true_ids = gt_map.get(sid, set())

    if not true_ids:
        continue

    evaluated += 1
    total_candidates += len(found)

    intersection = found & true_ids

    if intersection:
        hit_any += 1

    if true_ids.issubset(found):
        hit_all += 1


print("\n" + "=" * 70)
print("STAGE 2 PILOT RESULTS")
print("=" * 70)

print(f"Evaluated entities       : {evaluated:,}")
print(
    f"Average candidates/entity: "
    f"{total_candidates / max(evaluated, 1):.2f}"
)

print()
print(
    f"At-least-one recall      : "
    f"{hit_any / max(evaluated, 1):.4%}"
)

print(
    f"All-match recall         : "
    f"{hit_all / max(evaluated, 1):.4%}"
)

print()
print("=" * 70)
print("PILOT COMPLETE")
print("=" * 70)