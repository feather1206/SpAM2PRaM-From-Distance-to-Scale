# =========================================================
# Model Utilities
# =========================================================

import numpy as np
import pandas as pd

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor


from preprocessing import (
    get_all_items,
    make_train_test,
)


# =========================================================
# Ordered Model
# =========================================================

def fit_ordered_model(
    X_train,
    y_train,
    distr="logit",
):
    """
    Fit an OrderedModel.

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series
    distr : {"logit", "probit"}

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

def fit_linear(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model

def fit_lasso(
    X_train,
    y_train,
    alpha=0.01,
):

    model = Lasso(alpha=alpha)

    model.fit(X_train, y_train)

    return model

def fit_randomforest(
    X_train,
    y_train,
    random_state=0,
):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model

def fit_xgboost(
    X_train,
    y_train,
    random_state=0,
):

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model

def fit_knn(
    X_train,
    y_train,
    n_neighbors=5,
):

    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# One Monte Carlo Trial
# =========================================================

def run_one_trial(
    spam_pairs,
    pram,
    train_item_num=6,
    random_state=None,
):
    """
    Run one Monte Carlo trial.

    Parameters
    ----------
    spam_pairs : DataFrame
        Columns:
            A
            B
            Distance

    pram : DataFrame
        Columns:
            Subject
            A
            B
            Resp

    train_item_num : int

    random_state : int or None

    Returns
    -------
    dict or None
    """

    # -----------------------------------------------------
    # Random Seed
    # -----------------------------------------------------

    if random_state is not None:
        np.random.seed(random_state)

    # -----------------------------------------------------
    # Randomly choose products
    # -----------------------------------------------------

    items = get_all_items(spam_pairs)

    train_items = np.random.choice(
        items,
        size=train_item_num,
        replace=False,
    )

    # -----------------------------------------------------
    # Train/Test split
    # -----------------------------------------------------

    train_pairs, test_pairs = make_train_test(
        spam_pairs,
        train_items,
    )

    # -----------------------------------------------------
    # Merge PRaM (Train)
    # -----------------------------------------------------

    train_df = pd.merge(
        train_pairs,
        pram,
        on=["A", "B"],
        how="inner",
    )

    # -----------------------------------------------------
    # Merge PRaM (Test)
    # -----------------------------------------------------

    test_mean = (
        pram
        .groupby(["A", "B"], as_index=False)
        .agg(
            Resp_mean=("Resp", "mean")
        )
    )

    test_df = pd.merge(
        test_pairs,
        test_mean,
        on=["A", "B"],
        how="inner",
    )

    # -----------------------------------------------------
    # Prepare Data
    # -----------------------------------------------------

    X_train = train_df[["Distance"]]
    y_train = train_df["Resp"].astype(int)

    X_test = test_df[["Distance"]]
    y_true = test_df["Resp_mean"]

    # -----------------------------------------------------
    # Category Check
    # -----------------------------------------------------

    categories = np.sort(y_train.unique())

    if len(categories) < 2:
        return None

    # -----------------------------------------------------
    # Fit Models
    # -----------------------------------------------------

    models = {

        "Logit":
            fit_ordered_model(
                X_train,
                y_train,
                distr="logit",
            ),

        "Probit":
            fit_ordered_model(
                X_train,
                y_train,
                distr="probit",
            ),

        "Linear":
            fit_linear(
                X_train,
                y_train,
            ),

        "Lasso":
            fit_lasso(
                X_train,
                y_train,
            ),

        "RandomForest":
            fit_randomforest(
                X_train,
                y_train,
                random_state=random_state,
            ),

        "XGBoost":
            fit_xgboost(
                X_train,
                y_train,
                random_state=random_state,
            ),

        "KNN":
            fit_knn(
                X_train,
                y_train,
            ),
    }

    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return {
        "train_items": train_items,
        "train_df": train_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_true": y_true,
        "categories": categories,
        "models": models,
    }