import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

print("=" * 70)
print("FAST MULTI-PASS CANDIDATE GENERATOR")
print("=" * 70)

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MAX_CANDIDATES = 30

# ---------------------------------------------------------
# 1. LOAD CLEAN DATA
# ---------------------------------------------------------

print("\n[1/6] Loading clean data...")

s1 = pd.read_csv(
    'clean_test_data/clean_source1.tsv',
    sep='\t',
    usecols=['entity_id', 'clean_name', 'clean_address', 'clean_country']
)

s2 = pd.read_csv(
    'clean_test_data/clean_source2.tsv',
    sep='\t',
    usecols=['entity_id', 'clean_name', 'clean_address', 'clean_country']
)

s3 = pd.read_csv(
    'clean_test_data/clean_source3.tsv',
    sep='\t',
    usecols=['entity_id', 'clean_name', 'clean_address', 'clean_country']
)

target_pool = pd.concat([s2, s3], ignore_index=True)

print(f"Source 1: {len(s1):,}")
print(f"Target pool: {len(target_pool):,}")

# ---------------------------------------------------------
# 2. CONVERT TO NUMPY ARRAYS
# ---------------------------------------------------------

print("\n[2/6] Preparing fast lookup arrays...")

source_ids = s1['entity_id'].astype(str).to_numpy()

target_ids = target_pool['entity_id'].astype(str).to_numpy()
target_names = target_pool['clean_name'].fillna("").astype(str).to_numpy()
target_addresses = target_pool['clean_address'].fillna("").astype(str).to_numpy()
target_countries = target_pool['clean_country'].fillna("").astype(str).to_numpy()

source_names = s1['clean_name'].fillna("").astype(str).to_numpy()
source_addresses = s1['clean_address'].fillna("").astype(str).to_numpy()
source_countries = s1['clean_country'].fillna("").astype(str).to_numpy()

# ---------------------------------------------------------
# 3. BUILD EXACT-NAME INDEX
# ---------------------------------------------------------

print("\n[3/6] Building exact-name index...")

name_index = defaultdict(list)

for idx, name in enumerate(target_names):

    if name:
        name_index[name].append(idx)

print(f"Unique target names indexed: {len(name_index):,}")

# ---------------------------------------------------------
# 4. BUILD COMPACT-NAME INDEX
# ---------------------------------------------------------

print("\n[4/6] Building compact-name index...")

def compact(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

compact_index = defaultdict(list)

for idx, name in enumerate(target_names):

    if name:
        key = compact(name)

        if key:
            compact_index[key].append(idx)

print(f"Unique compact names indexed: {len(compact_index):,}")

# ---------------------------------------------------------
# 5. BUILD NAME TOKEN INDEX
# ---------------------------------------------------------

print("\n[5/6] Building token index...")

token_index = defaultdict(list)

for idx, name in enumerate(target_names):

    if not name:
        continue

    tokens = set(name.split())

    # Ignore extremely common/small tokens.
    useful_tokens = [
        token for token in tokens
        if len(token) >= 4
    ]

    # Only index up to 3 useful tokens per entity.
    useful_tokens = useful_tokens[:3]

    for token in useful_tokens:
        token_index[token].append(idx)

print(f"Name tokens indexed: {len(token_index):,}")

# ---------------------------------------------------------
# 6. GENERATE CANDIDATES
# ---------------------------------------------------------

print("\n[6/6] Generating candidates...")

results = []

for i in range(len(s1)):

    if i % 10000 == 0:
        print(
            f"Processing {i:,} / {len(s1):,} "
            f"({i / len(s1) * 100:.1f}%)"
        )

    candidate_indices = set()

    name = source_names[i]

    # -----------------------------------------------------
    # PASS 1: EXACT NORMALIZED NAME
    # -----------------------------------------------------

    if name:
        for idx in name_index.get(name, []):
            candidate_indices.add(idx)

    # -----------------------------------------------------
    # PASS 2: COMPACT NAME
    # -----------------------------------------------------

    if len(candidate_indices) < MAX_CANDIDATES:

        key = compact(name)

        if key:
            for idx in compact_index.get(key, []):
                candidate_indices.add(idx)

                if len(candidate_indices) >= MAX_CANDIDATES:
                    break

    # -----------------------------------------------------
    # PASS 3: NAME TOKEN BLOCKING
    # -----------------------------------------------------

    if len(candidate_indices) < MAX_CANDIDATES:

        tokens = set(name.split())

        useful_tokens = [
            token for token in tokens
            if len(token) >= 4
        ]

        # Use the longest/most informative tokens first.
        useful_tokens.sort(key=len, reverse=True)

        for token in useful_tokens[:3]:

            for idx in token_index.get(token, []):

                candidate_indices.add(idx)

                if len(candidate_indices) >= MAX_CANDIDATES:
                    break

            if len(candidate_indices) >= MAX_CANDIDATES:
                break

    # -----------------------------------------------------
    # CONVERT INDEX → ENTITY ID
    # -----------------------------------------------------

    if candidate_indices:

        candidate_indices = list(candidate_indices)

        candidate_ids = target_ids[candidate_indices]

        # Remove accidental duplicates.
        candidate_ids = list(dict.fromkeys(candidate_ids))

        candidate_ids = candidate_ids[:MAX_CANDIDATES]

        candidates_str = ",".join(candidate_ids)

    else:
        candidates_str = ""

    results.append({
        'source1_entity_id': source_ids[i],
        'candidate_entity_ids': candidates_str
    })

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

print("\nSaving candidate pairs...")

os.makedirs('output', exist_ok=True)

output_path = 'output/candidate_pairs.tsv'

pd.DataFrame(results).to_csv(
    output_path,
    sep='\t',
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Output: {output_path}")
print(f"Source 1 entities: {len(results):,}")

print("Candidate generation completed.")