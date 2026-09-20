"""Compare all models using subject-based Monte Carlo splits."""

import os

from experiment import run_performance_curve, summarize_performance
from pipeline import load_dataset


# Monte Carlo repetitions
N_TRIALS = 500

# Number of training subjects: 4, 5, ..., 15
TRAIN_SIZES = range(4, 16)

# Output directory
OUTPUT_DIR = "./results"


def main():
    print("Experiment Settings")
    print("-------------------")
    print("Trials      :", N_TRIALS)
    print("Train sizes :", list(TRAIN_SIZES))

    pram, spam = load_dataset()

    print("\nPRaM")
    print(pram.head().to_string(index=False))
    print("\nSpAM")
    print(spam.head().to_string(index=False))

    results = run_performance_curve(
        spam=spam,
        pram=pram,
        train_sizes=TRAIN_SIZES,
        n_trials=N_TRIALS,
    )

    print("\nMonte Carlo Complete")
    print(results.head(20).to_string(index=False))

    summary = summarize_performance(results)
    print("\nPerformance Summary")
    print(summary.round(4).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(
        OUTPUT_DIR,
        "subject_performance_curve.csv",
    )
    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "subject_performance_summary.csv",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved:", results_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
