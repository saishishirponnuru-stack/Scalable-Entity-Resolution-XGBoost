import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

TRAIN = "output/ml_training_features_v4.tsv"
MODEL = "output/xgb_best_v2.json"

features = [
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

print("Loading training features...")

df = pd.read_csv(
    TRAIN,
    sep="\t",
    usecols=features + ["is_match"]
)

print("Rows:", len(df))

model = XGBClassifier()
model.load_model(MODEL)

print("Predicting probabilities...")

p = model.predict_proba(df[features])[:, 1]
y = df["is_match"].astype(int).values

thresholds = [
    0.88,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999
]

print()
print("=" * 85)
print(f"{'Threshold':<12}{'Precision':<14}{'Recall':<14}{'F0.5':<14}{'Predicted %':<14}")
print("=" * 85)

best = None

for t in thresholds:

    pred = (p >= t).astype(int)

    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)

    f05 = (
        1.25 * precision * recall /
        (0.25 * precision + recall)
        if precision + recall > 0
        else 0
    )

    pct = pred.mean() * 100

    print(
        f"{t:<12.3f}"
        f"{precision:<14.4f}"
        f"{recall:<14.4f}"
        f"{f05:<14.4f}"
        f"{pct:<14.2f}"
    )

    if best is None or f05 > best[1]:
        best = (t, f05, precision, recall)

print("=" * 85)

print()
print("BEST THRESHOLD ON LABELED TRAINING FEATURES")
print("Threshold :", best[0])
print("F0.5      :", round(best[1], 4))
print("Precision :", round(best[2], 4))
print("Recall    :", round(best[3], 4))