# =========================================================
# evaluation.py
#
# Evaluation for subject-based Train/Test Split
#
# Test:
#     Mean SpAM distance across TEST subjects
#         ->
#     Predicted PRaM
#
# Compare:
#     Predicted PRaM
#         vs
#     Measured mean PRaM across TEST subjects
#
# Metrics:
#     Pearson r
#     Spearman rho
#     Kendall tau
#     RMSE
#     MAE
#     Accuracy ±1
#     R2
#     Bias
#     Range ratio
#     SD ratio
# =========================================================


import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# =========================================================
# Ordered Model Prediction
# =========================================================

def predict_ordered(
    model_result,
    X_test,
    categories,
):
    """
    Predict expected PRaM rating from
    Ordered Logit / Ordered Probit.

    The ordered model returns probabilities
    for each PRaM category.

    Expected rating:

        y_hat = sum(
            category * probability
        )
    """

    # -----------------------------------------------------
    # Category probabilities
    # -----------------------------------------------------

    probabilities = (
        model_result.model.predict(
            model_result.params,
            exog=X_test,
        )
    )

    probabilities = np.asarray(
        probabilities
    )


    # -----------------------------------------------------
    # Categories
    # -----------------------------------------------------

    categories = np.asarray(
        categories,
        dtype=float,
    )


    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if (
        probabilities.shape[1]
        !=
        len(categories)
    ):

        raise ValueError(
            "Number of predicted probability columns "
            "does not match number of PRaM categories."
        )


    # -----------------------------------------------------
    # Expected PRaM rating
    # -----------------------------------------------------

    y_pred = (
        probabilities
        @
        categories
    )


    return y_pred


# =========================================================
# Standard Model Prediction
# =========================================================

def predict_standard(
    model,
    X_test,
):
    """
    Prediction for:

        Linear
        Lasso
        Random Forest
        XGBoost
        KNN
    """

    y_pred = model.predict(
        X_test
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )


    return y_pred


# =========================================================
# Safe Pearson Correlation
# =========================================================

def safe_pearson(
    y_true,
    y_pred,
):
    """
    Calculate Pearson correlation safely.
    """

    if len(y_true) < 2:

        return np.nan


    if (
        np.std(y_true) == 0
        or
        np.std(y_pred) == 0
    ):

        return np.nan


    r, _ = pearsonr(
        y_true,
        y_pred,
    )


    return float(r)


# =========================================================
# Safe Spearman Correlation
# =========================================================

def safe_spearman(
    y_true,
    y_pred,
):
    """
    Calculate Spearman correlation safely.
    """

    if len(y_true) < 2:

        return np.nan


    if (
        np.std(y_true) == 0
        or
        np.std(y_pred) == 0
    ):

        return np.nan


    rho, _ = spearmanr(
        y_true,
        y_pred,
    )


    return float(rho)


# =========================================================
# Safe Kendall Correlation
# =========================================================

def safe_kendall(
    y_true,
    y_pred,
):
    """
    Calculate Kendall's tau safely.
    """

    if len(y_true) < 2:

        return np.nan


    tau, _ = kendalltau(
        y_true,
        y_pred,
    )


    return float(tau)


# =========================================================
# Metric Calculation
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate evaluation metrics.

    Metrics
    -------
    Pearson r
        Linear correspondence.

    Spearman rho
        Rank-order correspondence.

    Kendall tau
        Rank-order correspondence.

    RMSE
        Root mean squared prediction error.

    MAE
        Mean absolute prediction error.

    Accuracy ±1
        Proportion of stimulus pairs for which
        the absolute prediction error is
        less than or equal to 1 PRaM scale point.

    R2
        Coefficient of determination.

    Bias
        Mean prediction error.

        Positive:
            overprediction

        Negative:
            underprediction

    Range ratio
        Predicted range / observed range.

        Ideal value:
            1.0

    SD ratio
        Predicted SD / observed SD.

        Ideal value:
            1.0
    """

    # =====================================================
    # Convert to arrays
    # =====================================================

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )


    # =====================================================
    # Remove invalid values
    # =====================================================

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(y_pred)
    )


    y_true = y_true[
        valid
    ]

    y_pred = y_pred[
        valid
    ]


    if len(y_true) == 0:

        return None


    # =====================================================
    # Correlations
    # =====================================================

    pearson_r = safe_pearson(
        y_true,
        y_pred,
    )


    spearman_rho = safe_spearman(
        y_true,
        y_pred,
    )


    kendall_tau = safe_kendall(
        y_true,
        y_pred,
    )


    # =====================================================
    # Prediction Errors
    # =====================================================

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


    mae = mean_absolute_error(
        y_true,
        y_pred,
    )


    # =====================================================
    # Accuracy ±1
    #
    # Correct when:
    #
    # |Observed PRaM - Predicted PRaM| <= 1
    #
    # =====================================================

    accuracy_pm1 = np.mean(
        np.abs(
            y_true
            -
            y_pred
        )
        <= 1.0
    )


    # =====================================================
    # R2
    # =====================================================

    if len(y_true) >= 2:

        r2 = r2_score(
            y_true,
            y_pred,
        )

    else:

        r2 = np.nan


    # =====================================================
    # Bias
    # =====================================================

    bias = np.mean(
        y_pred
        -
        y_true
    )


    # =====================================================
    # Distribution Information
    # =====================================================

    true_mean = np.mean(
        y_true
    )


    pred_mean = np.mean(
        y_pred
    )


    # -----------------------------------------------------
    # SD
    # -----------------------------------------------------

    if len(y_true) >= 2:

        true_sd = np.std(
            y_true,
            ddof=1,
        )

        pred_sd = np.std(
            y_pred,
            ddof=1,
        )

    else:

        true_sd = np.nan

        pred_sd = np.nan


    # -----------------------------------------------------
    # Min / Max
    # -----------------------------------------------------

    true_min = np.min(
        y_true
    )


    true_max = np.max(
        y_true
    )


    pred_min = np.min(
        y_pred
    )


    pred_max = np.max(
        y_pred
    )


    # =====================================================
    # Range Compression
    # =====================================================

    true_range = (
        true_max
        -
        true_min
    )


    pred_range = (
        pred_max
        -
        pred_min
    )


    if true_range > 0:

        range_ratio = (
            pred_range
            /
            true_range
        )

    else:

        range_ratio = np.nan


    # =====================================================
    # SD Ratio
    # =====================================================

    if (
        np.isfinite(true_sd)
        and
        true_sd > 0
    ):

        sd_ratio = (
            pred_sd
            /
            true_sd
        )

    else:

        sd_ratio = np.nan


    # =====================================================
    # Return
    # =====================================================

    return {

        "Pearson_r":
            pearson_r,

        "Spearman_rho":
            spearman_rho,

        "Kendall_tau":
            kendall_tau,

        "RMSE":
            float(rmse),

        "MAE":
            float(mae),

        "Accuracy_pm1":
            float(accuracy_pm1),

        "R2":
            float(r2),

        "Bias":
            float(bias),

        "True_mean":
            float(true_mean),

        "Pred_mean":
            float(pred_mean),

        "True_SD":
            float(true_sd),

        "Pred_SD":
            float(pred_sd),

        "True_min":
            float(true_min),

        "True_max":
            float(true_max),

        "Pred_min":
            float(pred_min),

        "Pred_max":
            float(pred_max),

        "Range_ratio":
            float(range_ratio),

        "SD_ratio":
            float(sd_ratio),

        "N_pairs":
            int(
                len(y_true)
            ),
    }


# =========================================================
# Evaluate One Model
# =========================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_true,
    categories,
):
    """
    Evaluate one prediction model.
    """

    # =====================================================
    # Prediction
    # =====================================================

    if model_name in [
        "Logit",
        "Probit",
    ]:

        y_pred = predict_ordered(
            model,
            X_test,
            categories,
        )

    else:

        y_pred = predict_standard(
            model,
            X_test,
        )


    # =====================================================
    # Clip predictions to PRaM scale
    #
    # PRaM = 1-7
    # =====================================================

    y_pred = np.clip(
        y_pred,
        1.0,
        7.0,
    )


    # =====================================================
    # Metrics
    # =====================================================

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )


    return (
        metrics,
        y_pred,
    )


# =========================================================
# Evaluate All Models
# =========================================================

def evaluate_all_models(
    trial_result,
):
    """
    Evaluate all fitted models from one
    subject-based train/test trial.

    Parameters
    ----------
    trial_result : dict

        Output from model.run_one_trial()


    Returns
    -------
    results : pandas.DataFrame

        One row for each model.


    predictions : dict

        Prediction arrays for each model.
    """

    # =====================================================
    # Retrieve Data
    # =====================================================

    models = trial_result[
        "models"
    ]


    X_test = trial_result[
        "X_test"
    ]


    y_true = trial_result[
        "y_true"
    ]


    categories = trial_result[
        "categories"
    ]


    train_subjects = trial_result[
        "train_subjects"
    ]


    test_subjects = trial_result[
        "test_subjects"
    ]


    # =====================================================
    # Storage
    # =====================================================

    rows = []

    predictions = {}


    # =====================================================
    # Evaluate Each Model
    # =====================================================

    for model_name, model in models.items():

        try:

            metrics, y_pred = (
                evaluate_model(
                    model_name=model_name,
                    model=model,
                    X_test=X_test,
                    y_true=y_true,
                    categories=categories,
                )
            )


            # -------------------------------------------------
            # Invalid result
            # -------------------------------------------------

            if metrics is None:

                continue


            # -------------------------------------------------
            # Store predictions
            # -------------------------------------------------

            predictions[
                model_name
            ] = y_pred


            # -------------------------------------------------
            # Basic information
            # -------------------------------------------------

            row = {

                "Model":
                    model_name,

                "N_train_subjects":
                    len(
                        train_subjects
                    ),

                "N_test_subjects":
                    len(
                        test_subjects
                    ),
            }


            # -------------------------------------------------
            # Add metrics
            # -------------------------------------------------

            row.update(
                metrics
            )


            # -------------------------------------------------
            # Store
            # -------------------------------------------------

            rows.append(
                row
            )


        except Exception as e:

            print(
                f"[Evaluation Error] "
                f"{model_name}: {e}"
            )


    # =====================================================
    # Convert to DataFrame
    # =====================================================

    results = pd.DataFrame(
        rows
    )


    return (
        results,
        predictions,
    )


# =========================================================
# Create Prediction DataFrame
# =========================================================

def create_prediction_dataframe(
    trial_result,
    predictions,
):
    """
    Create pair-level prediction dataframe.

    Useful for:
        scatter plots
        prediction inspection
        CSV export

    Output example
    --------------
    A
    B
    Distance
    Resp_mean
    Logit
    Probit
    Linear
    Lasso
    RandomForest
    XGBoost
    KNN
    """

    # =====================================================
    # Test Data
    # =====================================================

    test_df = (
        trial_result[
            "test_df"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # Add Model Predictions
    # =====================================================

    for model_name, y_pred in predictions.items():

        test_df[
            model_name
        ] = y_pred


    return test_df