import os
import numpy as np
import pandas as pd

from statsmodels.miscmodels.ordinal_model import OrderedModel

from pipeline import load_dataset


# =========================================================
# Settings
# =========================================================

OUTPUT_DIR = "./results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

TRAIN_SUBJECT_NUM = 10

RANDOM_STATE = 42


# =========================================================
# 1. Load individual data
# =========================================================

pram, spam = load_dataset()


# =========================================================
# 2. Subject IDs
# =========================================================

spam["Subject"] = (
    spam["Subject"]
    .astype(int)
)

pram["Subject"] = (
    pram["Subject"]
    .astype(int)
)


# Subjects existing in BOTH datasets
spam_subjects = set(
    spam["Subject"].unique()
)

pram_subjects = set(
    pram["Subject"].unique()
)

subjects = sorted(
    spam_subjects
    &
    pram_subjects
)


# =========================================================
# 3. Split participants
#
# 10 participants -> Training
# Remaining 20    -> Test
# =========================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

train_subjects = sorted(
    rng.choice(
        subjects,
        size=TRAIN_SUBJECT_NUM,
        replace=False,
    )
)

train_subject_set = set(
    train_subjects
)

test_subjects = [
    subject
    for subject in subjects
    if subject not in train_subject_set
]


# =========================================================
# 4. TRAIN SpAM
#
# Mean SpAM across the 10 training participants
# =========================================================

train_spam = spam[
    spam["Subject"].isin(
        train_subjects
    )
].copy()


train_spam_mean = (
    train_spam
    .groupby(
        [
            "A",
            "B",
        ],
        as_index=False,
    )
    .agg(
        SpAM_mean=(
            "Distance",
            "mean",
        )
    )
)


# =========================================================
# 5. TRAIN PRaM
#
# Keep individual PRaM ratings
# =========================================================

train_pram = pram[
    pram["Subject"].isin(
        train_subjects
    )
][
    [
        "Subject",
        "A",
        "B",
        "Resp",
    ]
].copy()


# =========================================================
# 6. Training data
#
# Mean SpAM of 10 participants
#          ->
# Individual PRaM of 10 participants
# =========================================================

train_df = pd.merge(
    train_pram,
    train_spam_mean,
    on=[
        "A",
        "B",
    ],
    how="inner",
)


train_df = (
    train_df
    .dropna(
        subset=[
            "SpAM_mean",
            "Resp",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 7. X / y
# =========================================================

X_train = (
    train_df[
        ["SpAM_mean"]
    ]
    .rename(
        columns={
            "SpAM_mean":
                "Distance"
        }
    )
)


y_train = (
    train_df[
        "Resp"
    ]
    .astype(int)
)


# =========================================================
# 8. Ordered Logistic Regression
# =========================================================

model = OrderedModel(
    y_train,
    X_train,
    distr="logit",
)


model_result = model.fit(
    method="bfgs",
    disp=False,
)


categories = np.sort(
    y_train.unique()
)


# =========================================================
# 9. TEST SpAM
#
# Mean SpAM across the remaining 20 participants
# =========================================================

test_spam = spam[
    spam["Subject"].isin(
        test_subjects
    )
].copy()


test_spam_mean = (
    test_spam
    .groupby(
        [
            "A",
            "B",
        ],
        as_index=False,
    )
    .agg(
        SpAM_mean=(
            "Distance",
            "mean",
        )
    )
)


# =========================================================
# 10. Measured TEST PRaM
#
# Mean PRaM across the remaining 20 participants
# =========================================================

test_pram = pram[
    pram["Subject"].isin(
        test_subjects
    )
].copy()


test_pram_mean = (
    test_pram
    .groupby(
        [
            "A",
            "B",
        ],
        as_index=False,
    )
    .agg(
        PRaM_measured=(
            "Resp",
            "mean",
        )
    )
)


# =========================================================
# 11. Predict PRaM from TEST mean SpAM
# =========================================================

X_test = (
    test_spam_mean[
        ["SpAM_mean"]
    ]
    .rename(
        columns={
            "SpAM_mean":
                "Distance"
        }
    )
)


# Category probabilities
probs = model_result.model.predict(
    model_result.params,
    X_test,
)


# Expected PRaM rating
#
# P(1)*1 + P(2)*2 + ... + P(7)*7
#
predicted_pram = (
    probs
    @
    categories
)


# =========================================================
# 12. Prediction result
# =========================================================

prediction_df = test_spam_mean[
    [
        "A",
        "B",
        "SpAM_mean",
    ]
].copy()


prediction_df[
    "PRaM_predicted"
] = predicted_pram


# Add measured mean PRaM
prediction_df = pd.merge(
    prediction_df,
    test_pram_mean,
    on=[
        "A",
        "B",
    ],
    how="inner",
)


# =========================================================
# 13. Save prediction
# =========================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "PRaM_prediction.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 14. Save participant split
# =========================================================

split_df = pd.DataFrame(
    {
        "Subject": subjects,
        "Group": [
            "Train"
            if subject in train_subject_set
            else "Test"
            for subject in subjects
        ],
    }
)


split_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "subject_split.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 15. Print
# =========================================================

print(
    prediction_df
)

print()
print(
    "Training subjects:",
    train_subjects
)

print(
    "Test subjects:",
    test_subjects
)
