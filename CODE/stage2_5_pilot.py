import pandas as pd
import re
from collections import defaultdict
from rapidfuzz import fuzz

print("=" * 75)
print("STAGE 2.5 — CONTROLLED HIGH-RECALL PILOT")
print("=" * 75)

# ============================================================
# SETTINGS
# ============================================================

PILOT_ROWS = 10000

# We will test several candidate limits.
K_VALUES = [50, 75, 100, 150]

# Hard limits prevent common tokens/grams from exploding.
TOKEN_POSTING_LIMIT = 300
GRAM_POSTING_LIMIT = 300

# Maximum candidates allowed before ranking.
MAX_BROAD_CANDIDATES = 5000

S1_FILE = "clean_train_data/clean_source1.tsv"
S2_FILE = "clean_train_data/clean_source2.tsv"
S3_FILE = "clean_train_data/clean_source3.tsv"
GT_FILE = "train/train_ground_truth.tsv"


# ============================================================
# HELPERS
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def compact(x):
    return re.sub(r"[^a-z0-9]", "", norm(x))


def words(x):
    return set(re.findall(r"[a-z0-9]+", norm(x)))


def nums(x):
    return set(re.findall(r"\b\d+[a-z]?\b", norm(x)))


def chargrams(x, n=3):
    x = compact(x)

    if len(x) < n:
        return {x} if x else set()

    return {
        x[i:i+n]
        for i in range(len(x) - n + 1)
    }


# ============================================================
# 1. SOURCE 1
# ============================================================

print("\n[1/7] Loading Source 1 pilot...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    nrows=PILOT_ROWS
)

print(f"Source 1 rows: {len(s1):,}")


# ============================================================
# 2. TARGET DATA
# ============================================================

print("\n[2/7] Loading target pool...")

cols = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "clean_name",
    "clean_address",
    "clean_country"
]

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=cols
)

print(f"Source 2: {len(s2):,}")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=cols
)

print(f"Source 3: {len(s3):,}")

target = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(f"Total target pool: {len(target):,}")


# ============================================================
# 3. BUILD CONTROLLED INDEXES
# ============================================================

print("\n[3/7] Building controlled retrieval indexes...")

exact_index = defaultdict(list)
compact_index = defaultdict(list)
address_index = defaultdict(list)

# We use temporary posting lists.
token_index = defaultdict(list)
gram_index = defaultdict(list)

target_name = {}
target_addr = {}
target_country = {}


for i, row in target.iterrows():

    eid = row["entity_id"]

    name = norm(row["clean_name"])
    addr = norm(row["clean_address"])
    country = norm(row["clean_country"])

    target_name[eid] = name
    target_addr[eid] = addr
    target_country[eid] = country

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if name:
        exact_index[name].append(eid)

        c = compact(name)

        if c:
            compact_index[c].append(eid)

    # --------------------------------------------------------
    # Address number signature
    # --------------------------------------------------------

    nset = nums(addr)

    if nset:

        key = "|".join(sorted(nset))

        # Address lists are naturally much smaller.
        address_index[key].append(eid)

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    for token in words(name):

        if len(token) < 3:
            continue

        lst = token_index[token]

        if lst is not None:

            if len(lst) < TOKEN_POSTING_LIMIT:
                lst.append(eid)
            else:
                # Mark extremely common token as unusable.
                token_index[token] = None

    # --------------------------------------------------------
    # Character grams
    # --------------------------------------------------------

    for gram in list(chargrams(name))[:20]:

        lst = gram_index[gram]

        if lst is not None:

            if len(lst) < GRAM_POSTING_LIMIT:
                lst.append(eid)
            else:
                gram_index[gram] = None


# Remove exploded/common entries.

token_index = {
    k: v
    for k, v in token_index.items()
    if v is not None
}

gram_index = {
    k: v
    for k, v in gram_index.items()
    if v is not None
}

print(f"Exact names       : {len(exact_index):,}")
print(f"Compact names     : {len(compact_index):,}")
print(f"Addresses         : {len(address_index):,}")
print(f"Usable tokens     : {len(token_index):,}")
print(f"Usable char grams : {len(gram_index):,}")


# ============================================================
# 4. GROUND TRUTH
# ============================================================

print("\n[4/7] Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

pilot_ids = set(s1["entity_id"])

gt = gt[
    gt["source1_entity_id"].isin(pilot_ids)
]

gt_map = {}

for _, row in gt.iterrows():

    value = row.get("matched_entity_ids", "")

    if pd.isna(value):
        ids = set()
    else:
        ids = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_map[row["source1_entity_id"]] = ids

print(f"Ground-truth rows: {len(gt_map):,}")


# ============================================================
# 5. CONTROLLED RETRIEVAL
# ============================================================

print("\n[5/7] Running controlled retrieval...")

all_results = {}

retrieved_total = 0

broad_hit = 0
broad_all = 0
evaluated = 0

missed_broad = []


for count, (_, src) in enumerate(s1.iterrows()):

    sid = src["entity_id"]

    name = norm(src["clean_name"])
    addr = norm(src["clean_address"])
    country = norm(src["clean_country"])

    found = set()

    # --------------------------------------------------------
    # EXACT
    # --------------------------------------------------------

    if name:
        found.update(
            exact_index.get(name, [])
        )

    # --------------------------------------------------------
    # COMPACT
    # --------------------------------------------------------

    cname = compact(name)

    if cname:
        found.update(
            compact_index.get(cname, [])
        )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    nset = nums(addr)

    if nset:

        key = "|".join(sorted(nset))

        found.update(
            address_index.get(key, [])
        )

    # --------------------------------------------------------
    # TOKEN RETRIEVAL
    #
    # Only use the rarest available tokens.
    # --------------------------------------------------------

    source_tokens = [
        t for t in words(name)
        if len(t) >= 3 and t in token_index
    ]

    source_tokens.sort(
        key=lambda t: len(token_index[t])
    )

    for token in source_tokens[:3]:

        found.update(
            token_index[token]
        )

    # --------------------------------------------------------
    # CHARACTER RETRIEVAL
    #
    # Use rarest grams.
    # --------------------------------------------------------

    source_grams = [
        g for g in chargrams(name)
        if g in gram_index
    ]

    source_grams.sort(
        key=lambda g: len(gram_index[g])
    )

    for gram in source_grams[:10]:

        found.update(
            gram_index[gram]
        )

    # --------------------------------------------------------
    # COUNTRY FILTER
    # --------------------------------------------------------

    if country:

        same_country = {
            eid
            for eid in found
            if target_country.get(eid, "") == country
        }

        if same_country:
            found = same_country

    # --------------------------------------------------------
    # HARD SAFETY LIMIT
    # --------------------------------------------------------

    if len(found) > MAX_BROAD_CANDIDATES:

        # Rank by cheap name ratio first.
        rough = []

        for eid in found:

            score = fuzz.token_set_ratio(
                name,
                target_name.get(eid, "")
            )

            rough.append(
                (score, eid)
            )

        rough.sort(
            reverse=True
        )

        found = {
            eid
            for _, eid in rough[:MAX_BROAD_CANDIDATES]
        }

    retrieved_total += len(found)

    # --------------------------------------------------------
    # BROAD RECALL
    # --------------------------------------------------------

    true_ids = gt_map.get(sid, set())

    if true_ids:

        evaluated += 1

        if found & true_ids:
            broad_hit += 1
        else:

            if len(missed_broad) < 10:

                missed_broad.append(
                    (
                        sid,
                        true_ids,
                        len(found)
                    )
                )

        if true_ids.issubset(found):
            broad_all += 1

    # --------------------------------------------------------
    # SCORE CANDIDATES
    # --------------------------------------------------------

    scored = []

    src_nums = nset
    src_tokens = words(name)

    for eid in found:

        tname = target_name.get(eid, "")
        taddr = target_addr.get(eid, "")

        # NAME
        nr = fuzz.ratio(name, tname)
        nt = fuzz.token_set_ratio(name, tname)
        np = fuzz.partial_ratio(name, tname)

        name_score = (
            0.45 * nr +
            0.35 * nt +
            0.20 * np
        )

        # ADDRESS
        ar = fuzz.ratio(addr, taddr)
        at = fuzz.token_set_ratio(addr, taddr)

        addr_score = (
            0.60 * ar +
            0.40 * at
        )

        # ADDRESS NUMBERS
        tgt_nums = nums(taddr)

        if src_nums and tgt_nums:

            number_score = (
                len(src_nums & tgt_nums) /
                max(len(src_nums | tgt_nums), 1)
            ) * 100

        else:
            number_score = 0

        # NAME TOKENS
        tgt_tokens = words(tname)

        if src_tokens and tgt_tokens:

            token_score = (
                len(src_tokens & tgt_tokens) /
                max(len(src_tokens | tgt_tokens), 1)
            ) * 100

        else:
            token_score = 0

        # COUNTRY
        country_score = (
            100
            if country and
            target_country.get(eid, "") == country
            else 0
        )

        # FINAL RANK SCORE
        score = (
            0.45 * name_score +
            0.30 * addr_score +
            0.15 * number_score +
            0.07 * token_score +
            0.03 * country_score
        )

        scored.append(
            (score, eid)
        )

    scored.sort(
        reverse=True
    )

    # Save ranking.
    all_results[sid] = [
        eid for _, eid in scored
    ]

    if count % 500 == 0:

        print(
            f"Processed {count:,}/{len(s1):,} | "
            f"Broad avg: "
            f"{retrieved_total / (count + 1):.1f}"
        )


# ============================================================
# 6. EVALUATE K VALUES
# ============================================================

print("\n[6/7] Evaluating K values...")

print("\n" + "=" * 75)
print("STAGE 2.5 CONTROLLED RESULTS")
print("=" * 75)

print(
    f"Broad candidates/entity : "
    f"{retrieved_total / len(s1):.2f}"
)

print(
    f"Broad at-least-one recall: "
    f"{broad_hit / max(evaluated, 1):.4%}"
)

print(
    f"Broad all-match recall   : "
    f"{broad_all / max(evaluated, 1):.4%}"
)

print()

best_k = None
best_recall = -1

for k in K_VALUES:

    hit = 0
    all_hit = 0
    total = 0

    for sid, ranked in all_results.items():

        true_ids = gt_map.get(sid, set())

        if not true_ids:
            continue

        total += 1

        selected = set(
            ranked[:k]
        )

        if selected & true_ids:
            hit += 1

        if true_ids.issubset(selected):
            all_hit += 1

    recall = hit / max(total, 1)
    all_recall = all_hit / max(total, 1)

    print(
        f"K={k:3d} | "
        f"At-least-one: {recall:.4%} | "
        f"All-match: {all_recall:.4%}"
    )

    if recall > best_recall:

        best_recall = recall
        best_k = k


print()
print(
    f"Selected K based on pilot recall: {best_k}"
)


# ============================================================
# SAVE BEST PILOT
# ============================================================

output = []

for sid, ranked in all_results.items():

    output.append(
        {
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(
                ranked[:best_k]
            )
        }
    )

pd.DataFrame(output).to_csv(
    "output/stage2_5_best_pilot.tsv",
    sep="\t",
    index=False
)

print(
    "\nSaved: output/stage2_5_best_pilot.tsv"
)

print("=" * 75)
print("STAGE 2.5 COMPLETE")
print("=" * 75)