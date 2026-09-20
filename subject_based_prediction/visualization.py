# =========================================================
# visualization.py
#
# Visualization for subject-based Monte Carlo evaluation
#
# Expected result format:
#
# TrainSubjects
# Model
# Pearson_r
# Spearman_rho
# Kendall_tau
# RMSE
# MAE
# Accuracy_pm1
# R2
# Bias
# Range_ratio
# SD_ratio
# ...
# =========================================================


import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Utility:
# Model names
# =========================================================

def get_model_names(
    results
):
    """
    Get model names from result DataFrame.
    """

    if (
        "Model"
        not in
        results.columns
    ):

        return []

    return sorted(
        results[
            "Model"
        ]
        .dropna()
        .unique()
    )


# =========================================================
# Utility:
# Training subject sizes
# =========================================================

def get_train_sizes(
    results
):
    """
    Get training subject numbers.
    """

    if (
        "TrainSubjects"
        not in
        results.columns
    ):

        return []

    return sorted(
        results[
            "TrainSubjects"
        ]
        .dropna()
        .unique()
    )


# =========================================================
# Boxplot
# =========================================================

def plot_boxplot(
    results,
    metric="RMSE",
):
    """
    Boxplot for one metric
    across training subject sizes.

    One panel per model.
    """

    models = get_model_names(
        results
    )

    train_sizes = get_train_sizes(
        results
    )


    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------

    if len(models) == 0:

        print(
            "No models found."
        )

        return


    if metric not in results.columns:

        print(
            f"Metric not found: {metric}"
        )

        return


    # =====================================================
    # Figure
    # =====================================================

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(
            4 * len(models),
            5,
        ),
        sharey=True,
    )


    if len(models) == 1:

        axes = [
            axes
        ]


    # =====================================================
    # Model loop
    # =====================================================

    for ax, model in zip(
        axes,
        models,
    ):

        data = []

        labels = []


        for n in train_sizes:

            tmp = results[
                (
                    results[
                        "Model"
                    ]
                    ==
                    model
                )
                &
                (
                    results[
                        "TrainSubjects"
                    ]
                    ==
                    n
                )
            ][
                metric
            ].dropna()


            data.append(
                tmp.values
            )

            labels.append(
                str(n)
            )


        ax.boxplot(
            data
        )


        ax.set_title(
            model
        )


        ax.set_xlabel(
            "Training Subjects"
        )


        ax.set_xticklabels(
            labels
        )


    axes[0].set_ylabel(
        metric
    )


    plt.tight_layout()

    plt.show()


# =========================================================
# Histogram
# =========================================================

def plot_histogram(
    results,
    metric="RMSE",
    train_subject_num=None,
):
    """
    Histogram of one metric
    for each model.

    If train_subject_num is specified,
    only that training size is displayed.
    """

    models = get_model_names(
        results
    )


    if len(models) == 0:

        print(
            "No models found."
        )

        return


    if metric not in results.columns:

        print(
            f"Metric not found: {metric}"
        )

        return


    # =====================================================
    # Filter training size
    # =====================================================

    plot_data = results.copy()


    if train_subject_num is not None:

        plot_data = plot_data[
            plot_data[
                "TrainSubjects"
            ]
            ==
            train_subject_num
        ]


    # =====================================================
    # Figure
    # =====================================================

    n_models = len(
        models
    )


    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(
            7,
            3 * n_models,
        ),
    )


    if n_models == 1:

        axes = [
            axes
        ]


    # =====================================================
    # Model loop
    # =====================================================

    for ax, model in zip(
        axes,
        models,
    ):

        values = plot_data[
            plot_data[
                "Model"
            ]
            ==
            model
        ][
            metric
        ].dropna()


        ax.hist(
            values,
            bins=20,
        )


        ax.set_title(
            model
        )


        ax.set_xlabel(
            metric
        )


        ax.set_ylabel(
            "Count"
        )


    plt.tight_layout()

    plt.show()


# =========================================================
# Performance Curve
# =========================================================

def plot_performance_curve(
    results,
    metric="RMSE",
    ylabel=None,
):
    """
    Plot mean ± SD against
    number of training subjects.

    One line per model.
    """

    models = get_model_names(
        results
    )

    train_sizes = get_train_sizes(
        results
    )


    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------

    if len(models) == 0:

        print(
            "No models found."
        )

        return


    if metric not in results.columns:

        print(
            f"Metric not found: {metric}"
        )

        return


    if ylabel is None:

        ylabel = metric


    # =====================================================
    # Figure
    # =====================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )


    # =====================================================
    # Model loop
    # =====================================================

    for model in models:

        means = []

        stds = []


        for n in train_sizes:

            tmp = results[
                (
                    results[
                        "Model"
                    ]
                    ==
                    model
                )
                &
                (
                    results[
                        "TrainSubjects"
                    ]
                    ==
                    n
                )
            ][
                metric
            ].dropna()


            if len(tmp) == 0:

                means.append(
                    np.nan
                )

                stds.append(
                    np.nan
                )

                continue


            means.append(
                tmp.mean()
            )


            stds.append(
                tmp.std()
            )


        means = np.asarray(
            means,
            dtype=float,
        )


        stds = np.asarray(
            stds,
            dtype=float,
        )


        plt.errorbar(
            train_sizes,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=model,
        )


    # =====================================================
    # Labels
    # =====================================================

    plt.xlabel(
        "Number of Training Subjects"
    )


    plt.ylabel(
        ylabel
    )


    plt.title(
        ylabel
    )


    plt.xticks(
        train_sizes
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()


    plt.tight_layout()

    plt.show()


# =========================================================
# Plot All Main Metrics
# =========================================================

def plot_all_performance_curves(
    results
):
    """
    Plot all main evaluation metrics.

    Metrics:
        RMSE
        MAE
        Accuracy ±1
        R²
        Pearson r
        Spearman rho
        Kendall tau
        Range ratio
        SD ratio
    """

    metrics = [

        (
            "RMSE",
            "RMSE",
        ),

        (
            "MAE",
            "MAE",
        ),

        (
            "Accuracy_pm1",
            "Accuracy ±1",
        ),

        (
            "R2",
            "R²",
        ),

        (
            "Pearson_r",
            "Pearson r",
        ),

        (
            "Spearman_rho",
            "Spearman ρ",
        ),

        (
            "Kendall_tau",
            "Kendall τ",
        ),

        (
            "Range_ratio",
            "Prediction / Observed Range",
        ),

        (
            "SD_ratio",
            "Prediction / Observed SD",
        ),

    ]


    for metric, ylabel in metrics:

        if (
            metric
            not in
            results.columns
        ):

            continue


        plot_performance_curve(
            results=results,
            metric=metric,
            ylabel=ylabel,
        )


# =========================================================
# Accuracy ±1 Plot
# =========================================================

def plot_accuracy_pm1(
    results
):
    """
    Plot Accuracy ±1 against
    number of training subjects.

    Accuracy ±1:
        proportion of stimulus pairs with

        |Measured PRaM - Predicted PRaM| <= 1
    """

    if (
        "Accuracy_pm1"
        not in
        results.columns
    ):

        print(
            "Accuracy_pm1 not found."
        )

        return


    plot_performance_curve(
        results=results,
        metric="Accuracy_pm1",
        ylabel="Accuracy ±1",
    )


# =========================================================
# Range Compression Plot
# =========================================================

def plot_range_ratio(
    results
):
    """
    Plot prediction range ratio.

    Ideal value = 1.

    < 1:
        prediction range is compressed

    > 1:
        prediction range is too wide
    """

    models = get_model_names(
        results
    )

    train_sizes = get_train_sizes(
        results
    )


    if (
        "Range_ratio"
        not in
        results.columns
    ):

        print(
            "Range_ratio not found."
        )

        return


    plt.figure(
        figsize=(
            8,
            6,
        )
    )


    for model in models:

        means = []

        stds = []


        for n in train_sizes:

            tmp = results[
                (
                    results[
                        "Model"
                    ]
                    ==
                    model
                )
                &
                (
                    results[
                        "TrainSubjects"
                    ]
                    ==
                    n
                )
            ][
                "Range_ratio"
            ].dropna()


            if len(tmp) == 0:

                means.append(
                    np.nan
                )

                stds.append(
                    np.nan
                )

                continue


            means.append(
                tmp.mean()
            )


            stds.append(
                tmp.std()
            )


        means = np.asarray(
            means,
            dtype=float,
        )

        stds = np.asarray(
            stds,
            dtype=float,
        )


        plt.errorbar(
            train_sizes,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=model,
        )


    # -----------------------------------------------------
    # Ideal line
    # -----------------------------------------------------

    plt.axhline(
        y=1.0,
        linestyle="--",
    )


    plt.xlabel(
        "Number of Training Subjects"
    )


    plt.ylabel(
        "Prediction / Observed Range"
    )


    plt.title(
        "Prediction Range Compression"
    )


    plt.xticks(
        train_sizes
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()


    plt.tight_layout()

    plt.show()


# =========================================================
# SD Ratio Plot
# =========================================================

def plot_sd_ratio(
    results
):
    """
    Plot prediction SD ratio.

    Ideal value = 1.

    < 1:
        predicted values are less variable
        than measured values

    > 1:
        predicted values are more variable
        than measured values
    """

    models = get_model_names(
        results
    )

    train_sizes = get_train_sizes(
        results
    )


    if (
        "SD_ratio"
        not in
        results.columns
    ):

        print(
            "SD_ratio not found."
        )

        return


    plt.figure(
        figsize=(
            8,
            6,
        )
    )


    for model in models:

        means = []

        stds = []


        for n in train_sizes:

            tmp = results[
                (
                    results[
                        "Model"
                    ]
                    ==
                    model
                )
                &
                (
                    results[
                        "TrainSubjects"
                    ]
                    ==
                    n
                )
            ][
                "SD_ratio"
            ].dropna()


            if len(tmp) == 0:

                means.append(
                    np.nan
                )

                stds.append(
                    np.nan
                )

                continue


            means.append(
                tmp.mean()
            )


            stds.append(
                tmp.std()
            )


        means = np.asarray(
            means,
            dtype=float,
        )

        stds = np.asarray(
            stds,
            dtype=float,
        )


        plt.errorbar(
            train_sizes,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=model,
        )


    # -----------------------------------------------------
    # Ideal line
    # -----------------------------------------------------

    plt.axhline(
        y=1.0,
        linestyle="--",
    )


    plt.xlabel(
        "Number of Training Subjects"
    )


    plt.ylabel(
        "Prediction / Observed SD"
    )


    plt.title(
        "Prediction Variability"
    )


    plt.xticks(
        train_sizes
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()


    plt.tight_layout()

    plt.show()


# =========================================================
# Prediction Scatter Plot
# =========================================================

def plot_prediction_scatter(
    prediction_df,
    model_name,
):
    """
    Plot measured mean PRaM
    against predicted PRaM.

    prediction_df must contain:

        Resp_mean
        <model_name>
    """

    # =====================================================
    # Check
    # =====================================================

    if (
        "Resp_mean"
        not in
        prediction_df.columns
    ):

        raise ValueError(
            "Resp_mean column not found."
        )


    if (
        model_name
        not in
        prediction_df.columns
    ):

        raise ValueError(
            f"{model_name} column not found."
        )


    # =====================================================
    # Data
    # =====================================================

    x = prediction_df[
        "Resp_mean"
    ].to_numpy(
        dtype=float
    )


    y = prediction_df[
        model_name
    ].to_numpy(
        dtype=float
    )


    # -----------------------------------------------------
    # Remove invalid values
    # -----------------------------------------------------

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    x = x[
        valid
    ]

    y = y[
        valid
    ]


    if len(x) == 0:

        print(
            "No valid prediction data."
        )

        return


    # =====================================================
    # Scatter
    # =====================================================

    plt.figure(
        figsize=(
            7,
            7,
        )
    )


    plt.scatter(
        x,
        y,
        alpha=0.7,
    )


    # =====================================================
    # Perfect agreement line
    # =====================================================

    lower = min(
        np.min(x),
        np.min(y),
    )


    upper = max(
        np.max(x),
        np.max(y),
    )


    plt.plot(
        [
            lower,
            upper,
        ],
        [
            lower,
            upper,
        ],
        linestyle="--",
        label="Perfect agreement",
    )


    # =====================================================
    # Regression line
    # =====================================================

    if (
        len(x) >= 2
        and
        np.std(x) > 0
    ):

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )


        x_line = np.linspace(
            lower,
            upper,
            100,
        )


        y_line = (
            slope
            *
            x_line
            +
            intercept
        )


        plt.plot(
            x_line,
            y_line,
            label=(
                f"Prediction trend "
                f"(slope={slope:.3f})"
            ),
        )


    # =====================================================
    # Appearance
    # =====================================================

    plt.xlabel(
        "Measured Mean PRaM"
    )


    plt.ylabel(
        "Predicted PRaM"
    )


    plt.title(
        f"{model_name}: "
        f"Measured vs Predicted PRaM"
    )


    plt.xlim(
        lower,
        upper,
    )


    plt.ylim(
        lower,
        upper,
    )


    plt.gca().set_aspect(
        "equal",
        adjustable="box",
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()


    plt.tight_layout()

    plt.show()


# =========================================================
# Ordered Logit Summary Plot
# =========================================================

def plot_ordered_logit_summary(
    results
):
    """
    Convenience function for the main Ordered Logit model.

    Plots:
        Spearman rho
        RMSE
        MAE
        R2
        Accuracy ±1
        Range ratio
        SD ratio
    """

    if (
        "Model"
        not in
        results.columns
    ):

        print(
            "Model column not found."
        )

        return


    logit_results = results[
        results[
            "Model"
        ]
        ==
        "Logit"
    ].copy()


    if len(
        logit_results
    ) == 0:

        print(
            "Logit results not found."
        )

        return


    metrics = [

        (
            "Spearman_rho",
            "Spearman ρ",
        ),

        (
            "RMSE",
            "RMSE",
        ),

        (
            "MAE",
            "MAE",
        ),

        (
            "R2",
            "R²",
        ),

        (
            "Accuracy_pm1",
            "Accuracy ±1",
        ),

        (
            "Range_ratio",
            "Prediction / Observed Range",
        ),

        (
            "SD_ratio",
            "Prediction / Observed SD",
        ),

    ]


    for metric, ylabel in metrics:

        if (
            metric
            not in
            logit_results.columns
        ):

            continue


        plot_performance_curve(
            results=logit_results,
            metric=metric,
            ylabel=ylabel,
        )