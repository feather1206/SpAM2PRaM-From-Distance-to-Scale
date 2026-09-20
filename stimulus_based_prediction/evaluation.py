import numpy as np

from scipy.stats import spearmanr, kendalltau

from sklearn.metrics import r2_score


# =========================================================
# Metric Calculation
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
    categories,
    pair_result,
):
    """
    Calculate evaluation metrics.

    Evaluation is performed using continuous PRaM mean scores.
    """

    # ----------------------------------------
    # Rank correlation
    # ----------------------------------------

    rho, _ = spearmanr(
        y_true,
        y_pred,
    )

    tau, _ = kendalltau(
        y_true,
        y_pred,
    )


    # ----------------------------------------
    # Prediction error
    # ----------------------------------------

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )


    # ----------------------------------------
    # Prediction within one Likert point
    # ----------------------------------------

    accuracy_pm1 = np.mean(
        np.abs(
            y_true - y_pred
        ) <= 1
    )


    # ----------------------------------------
    # Explained variance
    # ----------------------------------------

    r2 = r2_score(
        y_true,
        y_pred,
    )


    # ----------------------------------------
    # Pair-level results
    # ----------------------------------------

    pair_result["True"] = y_true.values

    pair_result["Predicted"] = y_pred

    pair_result["Error"] = (
        y_pred
        -
        y_true.values
    )

    pair_result["AbsError"] = np.abs(
        pair_result["Error"]
    )


    return {

        "rho": rho,

        "tau": tau,

        "rmse": rmse,

        "mae": mae,

        "accuracy_pm1": accuracy_pm1,

        "r2": r2,

        "prediction": y_pred,

        "pair_result": pair_result,
    }


# =========================================================
# Ordered Model
# =========================================================

def evaluate_ordered(
    model_result,
    test_df,
    categories,
):

    X_test = test_df[
        ["Distance"]
    ]

    y_true = test_df[
        "Resp_mean"
    ]


    # Predict category probabilities
    probs = model_result.model.predict(
        model_result.params,
        X_test,
    )


    # Expected value of Likert categories
    y_pred = probs @ categories


    pair_result = test_df[
        [
            "A",
            "B",
            "Distance",
        ]
    ].copy()


    return calculate_metrics(
        y_true,
        y_pred,
        categories,
        pair_result,
    )


# =========================================================
# Regression Model
# =========================================================

def evaluate_regression(
    model,
    test_df,
    categories,
):

    X_test = test_df[
        ["Distance"]
    ]

    y_true = test_df[
        "Resp_mean"
    ]


    y_pred = model.predict(
        X_test
    )


    pair_result = test_df[
        [
            "A",
            "B",
            "Distance",
        ]
    ].copy()


    return calculate_metrics(
        y_true,
        y_pred,
        categories,
        pair_result,
    )


# =========================================================
# Evaluate All Models
# =========================================================

def evaluate_models(
    result_dict
):

    metrics = {}


    ordered_models = {

        "Logit",

        "Probit",

    }


    for name, model in result_dict["models"].items():


        if name in ordered_models:


            metrics[name] = evaluate_ordered(
                model,
                result_dict["test_df"],
                result_dict["categories"],
            )


        else:


            metrics[name] = evaluate_regression(
                model,
                result_dict["test_df"],
                result_dict["categories"],
            )


    return metrics


# =========================================================
# Print
# =========================================================

def print_result(metrics):

    for name, result in metrics.items():

        print("=" * 60)

        print(name)

        print("=" * 60)


        print(
            f"Spearman      : {result['rho']:.4f}"
        )


        print(
            f"Kendall       : {result['tau']:.4f}"
        )


        print(
            f"RMSE          : {result['rmse']:.4f}"
        )


        print(
            f"MAE           : {result['mae']:.4f}"
        )


        print(
            f"Accuracy ±1   : {result['accuracy_pm1']:.4f}"
        )


        print(
            f"R²            : {result['r2']:.4f}"
        )


        print()