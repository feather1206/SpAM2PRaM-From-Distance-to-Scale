import pandas as pd

from model import run_one_trial
from evaluation import evaluate_models


# =========================================================
# Monte Carlo
# =========================================================

def run_monte_carlo(
    spam_pairs,
    pram,
    n_trials=500,
    train_item_num=6,
):
    """
    Run Monte Carlo simulation.
    """

    results = []

    for trial in range(n_trials):

        result = run_one_trial(
            spam_pairs=spam_pairs,
            pram=pram,
            train_item_num=train_item_num,
            random_state=trial,
        )

        if result is None:
            continue

        metrics = evaluate_models(result)

        row = {
            "Trial": trial,
            "TrainItems": train_item_num,
        }

        # ----------------------------------------
        # Add all model results automatically
        # ----------------------------------------

        for model_name, metric in metrics.items():

            row[f"{model_name}_rho"] = metric["rho"]
            row[f"{model_name}_tau"] = metric["tau"]
            row[f"{model_name}_rmse"] = metric["rmse"]
            row[f"{model_name}_r2"] = metric["r2"]
            row[f"{model_name}_mae"] = metric["mae"]
            row[f"{model_name}_accuracy_pm1"] = metric["accuracy_pm1"]

        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# Performance Curve
# =========================================================

def run_performance_curve(
    spam_pairs,
    pram,
    train_sizes,
    n_trials=500,
):
    """
    Evaluate different numbers of training products.
    """

    all_results = []

    for train_item_num in train_sizes:

        print(f"Train Items = {train_item_num}")

        df = run_monte_carlo(
            spam_pairs=spam_pairs,
            pram=pram,
            n_trials=n_trials,
            train_item_num=train_item_num,
        )

        all_results.append(df)

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# =========================================================
# Summary
# =========================================================

def summarize_performance(results):
    """
    Summarize performance of every model.
    """

    summary = []

    train_sizes = sorted(
        results["TrainItems"].unique()
    )

    metric_names = [
        "rho",
        "tau",
        "rmse",
        "r2",
        "mae",
        "accuracy_pm1",
    ]

    model_names = sorted({

        c.split("_")[0]

        for c in results.columns

        if "_" in c

        and c != "TrainItems"

        and c != "Trial"

    })

    for train_size in train_sizes:

        df = results[
            results["TrainItems"] == train_size
        ]

        row = {
            "TrainItems": train_size
        }

        for model in model_names:

            for metric in metric_names:

                col = f"{model}_{metric}"

                if col not in df.columns:
                    continue

                row[f"{col}_mean"] = df[col].mean()
                row[f"{col}_sd"] = df[col].std()

                if metric in ["rmse", "mae"]:

                    row[f"{col}_best"] = df[col].min()

                else:

                    row[f"{col}_best"] = df[col].max()

        summary.append(row)

    return pd.DataFrame(summary)
