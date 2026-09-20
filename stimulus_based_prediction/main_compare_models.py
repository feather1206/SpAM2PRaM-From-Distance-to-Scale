# main_performance.py

from pathlib import Path

from config import OUTPUT_DIR, N_TRIALS

from pipeline import load_dataset

from experiment import (
    run_performance_curve,
    summarize_performance,
)

from visualization import (
    plot_performance_curve,
    plot_boxplot,
    plot_histogram,
)


def main():

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    pram, spam_pairs = load_dataset()

    print("Running Monte Carlo...")

    results = run_performance_curve(
        spam_pairs=spam_pairs,
        pram=pram,
        train_sizes=range(4, 9),
        n_trials=N_TRIALS,
    )

    results.to_csv(
        Path(OUTPUT_DIR) / "performance_curve.csv",
        index=False,
    )

    summary = summarize_performance(results)

    summary.to_csv(
        Path(OUTPUT_DIR) / "performance_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("Performance Summary")
    print("=" * 100)
    print(summary.round(4).to_string(index=False))

    plot_performance_curve(results)

    plot_boxplot(results)

    plot_histogram(results)


if __name__ == "__main__":
    main()
