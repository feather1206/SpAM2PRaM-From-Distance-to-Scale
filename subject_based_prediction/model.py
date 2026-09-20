# =========================================================
# model.py
#
# Subject-based Train/Test Split
#
# Train:
#     Mean SpAM distance across TRAIN subjects
#         ->
#     Individual PRaM ratings of TRAIN subjects
#
# Test:
#     Mean SpAM distance across TEST subjects
#         ->
#     Predicted mean PRaM
#
# Evaluation:
#     Predicted PRaM
#         vs
#     Measured mean PRaM of TEST subjects
# =========================================================


import numpy as np
import pandas as pd

from statsmodels.miscmodels.ordinal_model import OrderedModel

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
)

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor


# =========================================================
# Ordered Model
# =========================================================

def fit_ordered_model(
    X_train,
    y_train,
    distr="logit",
):
    """
    Fit ordinal regression model.

    Parameters
    ----------
    X_train : DataFrame
        Mean SpAM distance for each stimulus pair.

    y_train : Series
        Individual PRaM ratings (1-7).

    distr : str
        "logit" or "probit"

    Returns
    -------
    OrderedResults
    """

    model = OrderedModel(
        y_train,
        X_train,
        distr=distr,
    )

    result = model.fit(
        method="bfgs",
        disp=False,
    )

    return result


# =========================================================
# Linear Regression
# =========================================================

def fit_linear(
    X_train,
    y_train,
):
    """
    Linear regression.
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# Lasso
# =========================================================

def fit_lasso(
    X_train,
    y_train,
    alpha=0.01,
):
    """
    Lasso regression.
    """

    model = Lasso(
        alpha=alpha,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# Random Forest
# =========================================================

def fit_randomforest(
    X_train,
    y_train,
    random_state=0,
):
    """
    Random Forest regression.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# XGBoost
# =========================================================

def fit_xgboost(
    X_train,
    y_train,
    random_state=0,
):
    """
    XGBoost regression.
    """

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# KNN
# =========================================================

def fit_knn(
    X_train,
    y_train,
    n_neighbors=5,
):
    """
    K-nearest neighbors regression.
    """

    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# One Subject-Split Trial
# =========================================================

def run_one_trial(
    spam_data,
    pram,
    train_subject_num=6,
    random_state=None,
):
    """
    Run one subject-based train/test trial.

    ---------------------------------------------------------
    TRAIN
    ---------------------------------------------------------

    1. Randomly select training subjects.

    2. Average SpAM distances across training subjects
       for each stimulus pair.

    3. Keep PRaM ratings individually.

    4. Assign the SAME mean SpAM distance to each
       individual PRaM response for that pair.

       Mean SpAM distance
           ->
       Individual PRaM rating

    ---------------------------------------------------------
    TEST
    ---------------------------------------------------------

    1. Remaining subjects are used as test subjects.

    2. Average SpAM distances across test subjects.

    3. Average measured PRaM ratings across test subjects.

    4. Predict PRaM from mean SpAM distance.

    5. Compare predicted PRaM with measured mean PRaM.
    """

    # =====================================================
    # Copy data
    # =====================================================

    spam_data = spam_data.copy()
    pram = pram.copy()


    # =====================================================
    # Subject IDs -> int
    # =====================================================

    spam_data["Subject"] = (
        spam_data["Subject"]
        .astype(int)
    )

    pram["Subject"] = (
        pram["Subject"]
        .astype(int)
    )


    # =====================================================
    # Random generator
    # =====================================================

    rng = np.random.default_rng(
        random_state
    )


    # =====================================================
    # Subjects existing in BOTH datasets
    # =====================================================

    spam_subjects = set(
        int(s)
        for s in spam_data[
            "Subject"
        ].unique()
    )

    pram_subjects = set(
        int(s)
        for s in pram[
            "Subject"
        ].unique()
    )

    subjects = sorted(
        spam_subjects
        &
        pram_subjects
    )


    # =====================================================
    # Check subject number
    # =====================================================

    if len(subjects) <= train_subject_num:

        raise ValueError(
            "train_subject_num must be smaller "
            "than the total number of subjects."
        )

    if train_subject_num < 1:

        raise ValueError(
            "train_subject_num must be at least 1."
        )


    # =====================================================
    # Select training subjects
    # =====================================================

    train_subjects = rng.choice(
        subjects,
        size=train_subject_num,
        replace=False,
    )

    train_subjects = sorted(
        int(s)
        for s in train_subjects
    )


    # =====================================================
    # Remaining subjects = Test
    # =====================================================

    train_subject_set = set(
        train_subjects
    )

    test_subjects = [
        int(s)
        for s in subjects
        if s not in train_subject_set
    ]


    # =====================================================
    # TRAIN SpAM
    # =====================================================
    #
    # Select SpAM data only from training subjects.
    #
    # Then calculate mean distance for each pair.
    #
    # Result:
    #
    # A | B | Distance
    #
    # 1 | 2 | 420.5
    # 1 | 3 | 315.2
    # ...
    #
    # =====================================================

    train_spam = spam_data[
        spam_data["Subject"].isin(
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
            Distance=(
                "Distance",
                "mean",
            )
        )
    )


    # =====================================================
    # TRAIN PRaM
    # =====================================================
    #
    # PRaM is NOT averaged.
    #
    # Keep individual responses.
    #
    # =====================================================

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


    # =====================================================
    # Merge:
    #
    # Mean SpAM
    #     ->
    # Individual PRaM
    #
    # =====================================================
    #
    # Example:
    #
    # Mean SpAM A-B = 420
    #
    # Subject 1:
    #     420 -> PRaM 6
    #
    # Subject 2:
    #     420 -> PRaM 5
    #
    # Subject 3:
    #     420 -> PRaM 7
    #
    # ...
    #
    # =====================================================

    train_df = pd.merge(
        train_pram,
        train_spam_mean,
        on=[
            "A",
            "B",
        ],
        how="inner",
    )


    # =====================================================
    # Remove missing values
    # =====================================================

    train_df = (
        train_df
        .dropna(
            subset=[
                "Distance",
                "Resp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(train_df) == 0:

        return None


    # =====================================================
    # Training variables
    # =====================================================

    X_train = train_df[
        [
            "Distance",
        ]
    ]

    y_train = train_df[
        "Resp"
    ].astype(int)


    # =====================================================
    # PRaM categories
    # =====================================================

    categories = np.sort(
        y_train.unique()
    )

    if len(categories) < 2:

        return None


    # =====================================================
    # TEST SpAM
    # =====================================================
    #
    # Test subjects are also averaged
    # for each stimulus pair.
    #
    # =====================================================

    test_spam = spam_data[
        spam_data["Subject"].isin(
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
            Distance=(
                "Distance",
                "mean",
            )
        )
    )


    # =====================================================
    # TEST PRaM
    # =====================================================
    #
    # Test PRaM is averaged only for final evaluation.
    #
    # =====================================================

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
            Resp_mean=(
                "Resp",
                "mean",
            )
        )
    )


    # =====================================================
    # Merge Test Data
    # =====================================================
    #
    # A | B | Distance | Resp_mean
    #
    # =====================================================

    test_df = pd.merge(
        test_spam_mean,
        test_pram_mean,
        on=[
            "A",
            "B",
        ],
        how="inner",
    )


    # =====================================================
    # Remove missing values
    # =====================================================

    test_df = (
        test_df
        .dropna(
            subset=[
                "Distance",
                "Resp_mean",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(test_df) == 0:

        return None


    # =====================================================
    # Test variables
    # =====================================================

    X_test = test_df[
        [
            "Distance",
        ]
    ]

    y_true = test_df[
        "Resp_mean"
    ]


    # =====================================================
    # Fit All Models
    # =====================================================

    model_random_state = (
        0
        if random_state is None
        else random_state
    )

    models = {

        # -------------------------------------------------
        # Ordered Logistic Regression
        # -------------------------------------------------

        "Logit":
            fit_ordered_model(
                X_train,
                y_train,
                distr="logit",
            ),

        # -------------------------------------------------
        # Ordered Probit Regression
        # -------------------------------------------------

        "Probit":
            fit_ordered_model(
                X_train,
                y_train,
                distr="probit",
            ),

        # -------------------------------------------------
        # Linear Regression
        # -------------------------------------------------

        "Linear":
            fit_linear(
                X_train,
                y_train,
            ),

        # -------------------------------------------------
        # Lasso
        # -------------------------------------------------

        "Lasso":
            fit_lasso(
                X_train,
                y_train,
            ),

        # -------------------------------------------------
        # Random Forest
        # -------------------------------------------------

        "RandomForest":
            fit_randomforest(
                X_train,
                y_train,
                random_state=model_random_state,
            ),

        # -------------------------------------------------
        # XGBoost
        # -------------------------------------------------

        "XGBoost":
            fit_xgboost(
                X_train,
                y_train,
                random_state=model_random_state,
            ),

        # -------------------------------------------------
        # KNN
        # -------------------------------------------------

        "KNN":
            fit_knn(
                X_train,
                y_train,
            ),
    }


    # =====================================================
    # Return
    # =====================================================

    return {

        # -------------------------------------------------
        # Subject split
        # -------------------------------------------------

        "train_subjects":
            train_subjects,

        "test_subjects":
            test_subjects,

        # -------------------------------------------------
        # Training data
        # -------------------------------------------------

        "train_df":
            train_df,

        "train_spam_mean":
            train_spam_mean,

        # -------------------------------------------------
        # Test data
        # -------------------------------------------------

        "test_df":
            test_df,

        "test_spam_mean":
            test_spam_mean,

        "test_pram_mean":
            test_pram_mean,

        # -------------------------------------------------
        # Variables
        # -------------------------------------------------

        "X_train":
            X_train,

        "y_train":
            y_train,

        "X_test":
            X_test,

        "y_true":
            y_true,

        # -------------------------------------------------
        # Categories
        # -------------------------------------------------

        "categories":
            categories,

        # -------------------------------------------------
        # Models
        # -------------------------------------------------

        "models":
            models,
    }