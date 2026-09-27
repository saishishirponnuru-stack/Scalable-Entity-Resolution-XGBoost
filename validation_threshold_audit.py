import pandas as pd
import numpy as np
import hashlib
from xgboost import XGBClassifier

TRAIN = "output/ml_training_features_v4.tsv"
MODEL = "output/xgb_best_v2.json"

FEATURES = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "addr_ratio",
    "addr_token_set",
    "addr_partial",
    "country_match",
    "missing_addr",
    "name_len_diff",
    "addr_len_diff"
]

print("Loading model...")
model = XGBClassifier()
model.load_model(MODEL)

print("Scanning training feature file in chunks...")

thresholds = [
    0.88,
    0.89,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99
]

tp = {t: 0 for t in thresholds}
fp = {t: 0 for t in thresholds}
fn = {t: 0 for t in thresholds}

validation_rows = 0
validation_positive = 0
validation_negative = 0

for chunk_no, df in enumerate(
    pd.read_csv(
        TRAIN,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "is_match": int
        },
        usecols=FEATURES + ["source1_entity_id", "is_match"],
        chunksize=100000
    ),
    1
):

    # Recreate the original deterministic 10% source-level validation split.
    source_ids = df["source1_entity_id"].astype(str)

    is_validation = source_ids.map(
        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 10 == 0
    )

    df = df[is_validation].copy()

    if len(df) == 0:
        continue

    y = df["is_match"].values.astype(int)

    p = model.predict_proba(df[FEATURES])[:, 1]

    validation_rows += len(df)
    validation_positive += int(y.sum())
    validation_negative += int((y == 0).sum())

    for t in thresholds:

        pred = p >= t

        tp[t] += int(((pred == True) & (y == 1)).sum())
        fp[t] += int(((pred == True) & (y == 0)).sum())
        fn[t] += int(((pred == False) & (y == 1)).sum())

    if chunk_no % 20 == 0:
        print(
            f"Processed chunks: {chunk_no} | "
            f"Validation rows: {validation_rows:,}"
        )

print()
print("=" * 90)
print("HELD-OUT VALIDATION THRESHOLD AUDIT")
print("=" * 90)

print("Validation rows:", f"{validation_rows:,}")
print("Validation positives:", f"{validation_positive:,}")
print("Validation negatives:", f"{validation_negative:,}")
print()

print(
    f"{'Threshold':<12}"
    f"{'Precision':<14}"
    f"{'Recall':<14}"
    f"{'F0.5':<14}"
    f"{'Predicted %':<14}"
)

print("-" * 90)

best = None

for t in thresholds:

    precision = (
        tp[t] / (tp[t] + fp[t])
        if tp[t] + fp[t] > 0
        else 0
    )

    recall = (
        tp[t] / (tp[t] + fn[t])
        if tp[t] + fn[t] > 0
        else 0
    )

    f05 = (
        1.25 * precision * recall /
        (0.25 * precision + recall)
        if precision + recall > 0
        else 0
    )

    predicted_pct = (
        (tp[t] + fp[t]) / validation_rows * 100
    )

    print(
        f"{t:<12.2f}"
        f"{precision:<14.4f}"
        f"{recall:<14.4f}"
        f"{f05:<14.4f}"
        f"{predicted_pct:<14.2f}"
    )

    if best is None or f05 > best[1]:
        best = (
            t,
            f05,
            precision,
            recall
        )

print("=" * 90)

print()
print("BEST HELD-OUT THRESHOLD")
print("Threshold :", best[0])
print("F0.5      :", round(best[1], 4))
print("Precision :", round(best[2], 4))
print("Recall    :", round(best[3], 4))