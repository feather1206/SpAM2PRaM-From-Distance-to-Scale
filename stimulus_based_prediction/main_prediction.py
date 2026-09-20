# main_prediction.py

from pathlib import Path
import pandas as pd

from config import OUTPUT_DIR

from pipeline import load_dataset

from model import run_one_trial
from evaluation import evaluate_models


def main():

    # =====================================================
    # Output folder
    # =====================================================

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Load data
    # =====================================================

    print("Loading dataset...")

    pram, spam_pairs = load_dataset()

    # =====================================================
    # Train model
    # =====================================================

    print("Training model...")

    result = run_one_trial(
        spam_pairs=spam_pairs,
        pram=pram,
        train_item_num=6,
        random_state=42,
    )

    metrics = evaluate_models(result)

    # =====================================================
    # Select prediction model
    # =====================================================

    # model_name = "Logit"
    # model_name = "Probit"
    model_name = "Linear"
    # model_name = "Lasso"
    # model_name = "RandomForest"
    # model_name = "XGBoost"
    # model_name = "KNN"

    # =====================================================
    # Mean PRaM (measured)
    # =====================================================

    pram_mean = (
        pram
        .groupby(["A", "B"], as_index=False)
        .agg(
            PRaM=("Resp", "mean")
        )
    )

    # =====================================================
    # Predicted pairs
    # =====================================================

    pred = metrics[model_name]["pair_result"][
        ["A", "B", "Predicted"]
    ].copy()

    pred = pred.rename(
        columns={
            "Predicted": "PRaM"
        }
    )

    pred["DataType"] = "Predicted"

    # =====================================================
    # Measured pairs
    # =====================================================

    train_pairs = (
        result["train_df"][
            ["A", "B"]
        ]
        .drop_duplicates()
    )

    measured = pd.merge(
        train_pairs,
        pram_mean,
        on=["A", "B"],
        how="left",
    )

    measured["DataType"] = "Measured"

    # =====================================================
    # Merge measured + predicted
    # =====================================================

    completed = pd.concat(
        [
            measured,
            pred,
        ],
        ignore_index=True,
    )

    completed = (
        completed
        .sort_values(["A", "B"])
        .reset_index(drop=True)
    )

    # =====================================================
    # Save
    # =====================================================

    output_file = (
        Path(OUTPUT_DIR)
        / "PRaM_completed.csv"
    )

    completed.to_csv(
        output_file,
        index=False,
    )

    # =====================================================
    # Display
    # =====================================================

    print()
    print("=" * 60)
    print("Completed PRaM Dataset")
    print("=" * 60)

    print(completed.head(15))

    print()
    print(f"Saved to:\n{output_file}")


if __name__ == "__main__":
    main()
