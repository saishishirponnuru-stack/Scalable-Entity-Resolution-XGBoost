import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score
)

import hashlib
import gc
import os


print("=" * 80)
print("XGBOOST HYPERPARAMETER OPTIMIZATION")
print("=" * 80)


FEATURE_FILE = "output/ml_training_features_v4.tsv"

BEST_MODEL_FILE = "output/xgb_best_v2.json"
BEST_THRESHOLD_FILE = "output/best_threshold_v2.txt"

CHUNK_SIZE = 100000

NEGATIVES_PER_SOURCE = 2

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


# ============================================================
# DETERMINISTIC SOURCE-LEVEL SPLIT
# ============================================================

def is_validation_source(sid):

    h = hashlib.md5(
        str(sid).encode("utf-8")
    ).hexdigest()

    return int(h[:8], 16) % 10 == 0


# ============================================================
# LOAD + REDUCE TRAINING DATA
# ============================================================

print("\n[1/5] Loading training data...")

train_parts = []
val_parts = []

reader = pd.read_csv(
    FEATURE_FILE,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "candidate_id": str,
        "is_match": np.int8
    },
    usecols=[
        "source1_entity_id",
        "candidate_id",
        *FEATURES,
        "is_match"
    ],
    chunksize=CHUNK_SIZE
)

total_rows = 0

for chunk_no, df in enumerate(reader, start=1):

    total_rows += len(df)

    val_mask = df["source1_entity_id"].map(
        is_validation_source
    )

    train_df = df.loc[~val_mask]
    val_df = df.loc[val_mask]

    # --------------------------------------------------------
    # Keep ALL positives.
    # Keep only strongest hard negatives.
    # --------------------------------------------------------

    def reduce_group(data):

        if len(data) == 0:
            return data

        positives = data[
            data["is_match"] == 1
        ]

        negatives = data[
            data["is_match"] == 0
        ]

        negatives = (
            negatives
            .groupby(
                "source1_entity_id",
                sort=False
            )
            .head(NEGATIVES_PER_SOURCE)
        )

        return pd.concat(
            [
                positives,
                negatives
            ],
            ignore_index=True
        )

    train_parts.append(
        reduce_group(train_df)
    )

    val_parts.append(
        reduce_group(val_df)
    )

    if chunk_no % 10 == 0:

        print(
            f"Processed "
            f"{total_rows:,} rows..."
        )


print("\nData loading complete.")


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

val_df = pd.concat(
    val_parts,
    ignore_index=True
)

del train_parts
del val_parts

gc.collect()


# ============================================================
# NUMPY MATRICES
# ============================================================

print("\n[2/5] Creating matrices...")

X_train = (
    train_df[FEATURES]
    .fillna(0)
    .astype(np.float32)
    .to_numpy()
)

y_train = (
    train_df["is_match"]
    .astype(np.int8)
    .to_numpy()
)

X_val = (
    val_df[FEATURES]
    .fillna(0)
    .astype(np.float32)
    .to_numpy()
)

y_val = (
    val_df["is_match"]
    .astype(np.int8)
    .to_numpy()
)

del train_df
del val_df

gc.collect()


print(
    f"Training matrix:   {X_train.shape}"
)

print(
    f"Validation matrix: {X_val.shape}"
)

print(
    f"Training positives: "
    f"{y_train.sum():,}"
)

print(
    f"Training negatives: "
    f"{(y_train == 0).sum():,}"
)


# ============================================================
# CLASS WEIGHT
# ============================================================

positive_count = max(
    int(y_train.sum()),
    1
)

negative_count = max(
    int((y_train == 0).sum()),
    1
)

scale_pos_weight = min(
    negative_count / positive_count,
    5.0
)

print(
    f"\nScale positive weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

experiments = [

    {
        "name": "MODEL_A",
        "max_depth": 6,
        "min_child_weight": 3,
        "learning_rate": 0.08,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0
    },

    {
        "name": "MODEL_B",
        "max_depth": 7,
        "min_child_weight": 3,
        "learning_rate": 0.06,
        "subsample": 0.90,
        "colsample_bytree": 0.95,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0
    },

    {
        "name": "MODEL_C",
        "max_depth": 8,
        "min_child_weight": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 3.0
    },

    {
        "name": "MODEL_D",
        "max_depth": 6,
        "min_child_weight": 1,
        "learning_rate": 0.05,
        "subsample": 0.95,
        "colsample_bytree": 1.00,
        "reg_alpha": 0.00,
        "reg_lambda": 2.0
    }
]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def evaluate_thresholds(
    probabilities,
    y_true
):

    best = None

    thresholds = np.arange(
        0.50,
        0.991,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f05 = fbeta_score(
            y_true,
            predictions,
            beta=0.5,
            zero_division=0
        )

        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f05": f05
        }

        if (
            best is None
            or f05 > best["f05"]
        ):
            best = result

    return best


# ============================================================
# TRAIN EXPERIMENTS
# ============================================================

print("\n[3/5] Running model experiments...")

experiment_results = []

global_best = None
global_best_model = None


for config in experiments:

    print("\n" + "-" * 80)

    print(
        f"TRAINING {config['name']}"
    )

    print("-" * 80)

    model = xgb.XGBClassifier(

        n_estimators=700,

        learning_rate=config[
            "learning_rate"
        ],

        max_depth=config[
            "max_depth"
        ],

        min_child_weight=config[
            "min_child_weight"
        ],

        subsample=config[
            "subsample"
        ],

        colsample_bytree=config[
            "colsample_bytree"
        ],

        reg_alpha=config[
            "reg_alpha"
        ],

        reg_lambda=config[
            "reg_lambda"
        ],

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        max_bin=256,

        scale_pos_weight=scale_pos_weight,

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_val, y_val)
        ],

        verbose=100
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    result = evaluate_thresholds(
        probabilities,
        y_val
    )

    result["model"] = config["name"]

    experiment_results.append(
        result
    )

    print("\nRESULT")

    print(
        f"Threshold : "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"F0.5      : "
        f"{result['f05']:.4f}"
    )

    if (
        global_best is None
        or result["f05"] > global_best["f05"]
    ):

        global_best = result
        global_best_model = model


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    f"{'Model':<12}"
    f"{'Threshold':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F0.5':>12}"
)

for result in sorted(
    experiment_results,
    key=lambda x: x["f05"],
    reverse=True
):

    print(
        f"{result['model']:<12}"
        f"{result['threshold']:>12.2f}"
        f"{result['precision']:>12.4f}"
        f"{result['recall']:>12.4f}"
        f"{result['f05']:>12.4f}"
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n[4/5] Saving best model...")

global_best_model.save_model(
    BEST_MODEL_FILE
)

with open(
    BEST_THRESHOLD_FILE,
    "w"
) as f:

    f.write(
        str(global_best["threshold"])
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n[5/5] FINAL BEST MODEL")
print("=" * 80)

print(
    f"Model      : "
    f"{global_best['model']}"
)

print(
    f"Threshold  : "
    f"{global_best['threshold']:.2f}"
)

print(
    f"Precision  : "
    f"{global_best['precision']:.4f}"
)

print(
    f"Recall     : "
    f"{global_best['recall']:.4f}"
)

print(
    f"F0.5       : "
    f"{global_best['f05']:.4f}"
)

print(
    f"\nSaved model:"
    f"\n{BEST_MODEL_FILE}"
)

print(
    f"\nSaved threshold:"
    f"\n{BEST_THRESHOLD_FILE}"
)

print("=" * 80)

print(
    "\nBaseline F0.5 = 0.9774"
)

print(
    f"New best F0.5 = "
    f"{global_best['f05']:.4f}"
)

if global_best["f05"] > 0.9774:

    print(
        "\n🔥 IMPROVEMENT FOUND!"
    )

else:

    print(
        "\nNo improvement over baseline."
    )