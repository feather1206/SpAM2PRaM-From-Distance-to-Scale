import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Utility
# =========================================================

def get_model_names(results):

    names = set()

    for col in results.columns:

        if col.endswith("_rmse"):

            names.add(col.replace("_rmse", ""))

    return sorted(names)


# =========================================================
# Boxplot
# =========================================================

def plot_boxplot(results):

    models = get_model_names(results)

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4 * len(models), 5),
        sharey=True,
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):

        data = []

        train_sizes = sorted(
            results["TrainItems"].unique()
        )

        for n in train_sizes:

            data.append(

                results.loc[
                    results["TrainItems"] == n,
                    f"{model}_rmse"
                ]

            )

        ax.boxplot(data)

        ax.set_title(model)

        ax.set_xlabel("Train")

        ax.set_xticklabels(train_sizes)

    axes[0].set_ylabel("RMSE")

    plt.tight_layout()

    plt.show()


# =========================================================
# Histogram
# =========================================================

def plot_histogram(results):

    models = get_model_names(results)

    n = len(models)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6, 3 * n),
    )

    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):

        ax.hist(
            results[f"{model}_rmse"],
            bins=20,
        )

        ax.set_title(model)

        ax.set_xlabel("RMSE")

        ax.set_ylabel("Count")

    plt.tight_layout()

    plt.show()


# =========================================================
# Performance Curve
# =========================================================

def plot_performance_curve(results):

    models = get_model_names(results)

    train_sizes = sorted(
        results["TrainItems"].unique()
    )

    metrics = [
        ("rmse", "RMSE"),
        ("r2", "R²"),
        ("accuracy_pm1", "Accuracy ±1"),
    ]

    for metric, ylabel in metrics:

        plt.figure(figsize=(7,5))

        for model in models:

            mean = []

            std = []

            for n in train_sizes:

                tmp = results.loc[
                    results["TrainItems"] == n,
                    f"{model}_{metric}"
                ]

                mean.append(tmp.mean())

                std.append(tmp.std())

            mean = np.array(mean)

            std = np.array(std)

            plt.errorbar(
                train_sizes,
                mean,
                yerr=std,
                marker="o",
                capsize=4,
                label=model,
            )

        plt.xlabel("Number of Training Products")

        plt.ylabel(ylabel)

        plt.title(ylabel)

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()
