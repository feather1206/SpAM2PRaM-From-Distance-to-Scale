# =========================================================
# experiment.py
#
# Subject-based Monte Carlo Evaluation
#
# Train:
#     Mean SpAM across TRAIN subjects
#         ->
#     Individual PRaM ratings of TRAIN subjects
#
# Test:
#     Mean SpAM across TEST subjects
#         ->
#     Predicted PRaM
#
# Evaluation:
#     Predicted PRaM
#         vs
#     Measured mean PRaM of TEST subjects
# =========================================================


import pandas as pd

from model import run_one_trial

from evaluation import (
    evaluate_all_models,
)


# =========================================================
# One Training-Subject Size
# =========================================================

def run_monte_carlo(
    spam,
    pram,
    n_trials=500,
    train_subject_num=6,
):
    """
    Run repeated subject-based Monte Carlo trials.

    Parameters
    ----------
    spam : pandas.DataFrame

        Individual-level SpAM data.

        Columns:
            Subject
            A
            B
            Distance


    pram : pandas.DataFrame

        Individual-level PRaM data.

        Columns:
            Subject
            A
            B
            Resp


    n_trials : int

        Number of Monte Carlo repetitions.


    train_subject_num : int

        Number of subjects randomly selected
        for model training.


    Returns
    -------
    pandas.DataFrame

        One row per model per Monte Carlo trial.
    """

    # =====================================================
    # Storage
    # =====================================================

    all_rows = []


    # =====================================================
    # Monte Carlo loop
    # =====================================================

    for trial in range(
        n_trials
    ):

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        print(
            f"\r"
            f"Train subjects: "
            f"{train_subject_num:2d} | "
            f"Trial: "
            f"{trial + 1:4d}/{n_trials}",
            end="",
            flush=True,
        )


        # =================================================
        # One subject split
        # =================================================

        try:

            trial_result = run_one_trial(
                spam_data=spam,
                pram=pram,
                train_subject_num=train_subject_num,
                random_state=trial,
            )

        except Exception as e:

            print()

            print(
                f"[Trial Error] "
                f"Train={train_subject_num}, "
                f"Trial={trial}: "
                f"{e}"
            )

            continue


        # -------------------------------------------------
        # Invalid trial
        # -------------------------------------------------

        if trial_result is None:

            continue


        # =================================================
        # Evaluate all models
        # =================================================

        try:

            model_results, _ = (
                evaluate_all_models(
                    trial_result
                )
            )

        except Exception as e:

            print()

            print(
                f"[Evaluation Error] "
                f"Train={train_subject_num}, "
                f"Trial={trial}: "
                f"{e}"
            )

            continue


        # -------------------------------------------------
        # No valid model
        # -------------------------------------------------

        if len(
            model_results
        ) == 0:

            continue


        # =================================================
        # Add trial information
        # =================================================

        model_results[
            "Trial"
        ] = trial


        model_results[
            "TrainSubjects"
        ] = train_subject_num


        model_results[
            "TestSubjects"
        ] = len(
            trial_result[
                "test_subjects"
            ]
        )


        # -------------------------------------------------
        # Optional:
        # Store actual selected training subjects
        # -------------------------------------------------

        model_results[
            "TrainSubjectIDs"
        ] = ",".join(
            str(x)
            for x in trial_result[
                "train_subjects"
            ]
        )


        # =================================================
        # Store
        # =================================================

        all_rows.append(
            model_results
        )


    # =====================================================
    # Progress newline
    # =====================================================

    print()


    # =====================================================
    # Combine
    # =====================================================

    if len(
        all_rows
    ) == 0:

        return pd.DataFrame()


    results = pd.concat(
        all_rows,
        ignore_index=True,
    )


    return results


# =========================================================
# Performance Curve
# =========================================================

def run_performance_curve(
    spam,
    pram,
    train_sizes,
    n_trials=500,
):
    """
    Run Monte Carlo evaluation for several
    numbers of training subjects.

    Example
    -------
    train_sizes = range(4, 9)

    evaluates:

        4 subjects
        5 subjects
        6 subjects
        7 subjects
        8 subjects
    """

    # =====================================================
    # Storage
    # =====================================================

    all_results = []


    # =====================================================
    # Loop through training-set sizes
    # =====================================================

    for train_subject_num in train_sizes:

        print()

        print(
            "=" * 70
        )

        print(
            f"Training Subjects = "
            f"{train_subject_num}"
        )

        print(
            "=" * 70
        )


        # -------------------------------------------------
        # Monte Carlo
        # -------------------------------------------------

        df = run_monte_carlo(
            spam=spam,
            pram=pram,
            n_trials=n_trials,
            train_subject_num=train_subject_num,
        )


        # -------------------------------------------------
        # Skip empty result
        # -------------------------------------------------

        if len(
            df
        ) == 0:

            continue


        all_results.append(
            df
        )


    # =====================================================
    # Combine
    # =====================================================

    if len(
        all_results
    ) == 0:

        return pd.DataFrame()


    results = pd.concat(
        all_results,
        ignore_index=True,
    )


    return results


# =========================================================
# Summary
# =========================================================

def summarize_performance(
    results
):
    """
    Calculate mean, SD, and best value
    for every model and training-subject size.

    Returns
    -------
    pandas.DataFrame

        One row per:

            TrainSubjects × Model
    """

    # =====================================================
    # Empty check
    # =====================================================

    if len(
        results
    ) == 0:

        return pd.DataFrame()


    # =====================================================
    # Metrics
    # =====================================================

    metrics = [

        "Pearson_r",

        "Spearman_rho",

        "Kendall_tau",

        "RMSE",

        "MAE",

        "Accuracy_pm1",

        "R2",

        "Bias",

        "True_mean",

        "Pred_mean",

        "True_SD",

        "Pred_SD",

        "Range_ratio",

        "SD_ratio",

    ]


    # =====================================================
    # Summary rows
    # =====================================================

    rows = []


    # =====================================================
    # Group by training size and model
    # =====================================================

    grouped = results.groupby(
        [
            "TrainSubjects",
            "Model",
        ],
        sort=True,
    )


    for (
        train_subject_num,
        model_name,
    ), df in grouped:

        row = {

            "TrainSubjects":
                int(
                    train_subject_num
                ),

            "Model":
                model_name,

            "N_trials":
                int(
                    len(df)
                ),

        }


        # =================================================
        # Test subject number
        # =================================================

        if (
            "TestSubjects"
            in
            df.columns
        ):

            row[
                "TestSubjects"
            ] = int(
                df[
                    "TestSubjects"
                ].iloc[0]
            )


        # =================================================
        # Metrics
        # =================================================

        for metric in metrics:

            if metric not in df.columns:

                continue


            values = df[
                metric
            ]


            # ---------------------------------------------
            # Mean
            # ---------------------------------------------

            row[
                f"{metric}_mean"
            ] = values.mean()


            # ---------------------------------------------
            # SD
            # ---------------------------------------------

            row[
                f"{metric}_sd"
            ] = values.std()


            # ---------------------------------------------
            # Best
            # ---------------------------------------------

            # Lower is better
            if metric in [
                "RMSE",
                "MAE",
            ]:

                row[
                    f"{metric}_best"
                ] = values.min()


            # Bias:
            # closest to zero is best
            elif metric == "Bias":

                if len(
                    values.dropna()
                ) > 0:

                    best_index = (
                        values.abs()
                        .idxmin()
                    )

                    row[
                        f"{metric}_best"
                    ] = values.loc[
                        best_index
                    ]

                else:

                    row[
                        f"{metric}_best"
                    ] = float(
                        "nan"
                    )


            # Range ratio / SD ratio:
            # closest to 1 is best
            elif metric in [
                "Range_ratio",
                "SD_ratio",
            ]:

                valid_values = (
                    values
                    .dropna()
                )

                if len(
                    valid_values
                ) > 0:

                    best_index = (
                        (
                            valid_values
                            -
                            1.0
                        )
                        .abs()
                        .idxmin()
                    )

                    row[
                        f"{metric}_best"
                    ] = values.loc[
                        best_index
                    ]

                else:

                    row[
                        f"{metric}_best"
                    ] = float(
                        "nan"
                    )


            # Higher is better
            else:

                row[
                    f"{metric}_best"
                ] = values.max()


        rows.append(
            row
        )


    # =====================================================
    # DataFrame
    # =====================================================

    summary = pd.DataFrame(
        rows
    )


    # =====================================================
    # Sort
    # =====================================================

    summary = (
        summary
        .sort_values(
            [
                "TrainSubjects",
                "Model",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return summary


# =========================================================
# Wide Summary
# =========================================================

def summarize_performance_wide(
    results
):
    """
    Optional wide-format summary.

    Example columns:

        Logit_RMSE_mean
        Logit_RMSE_sd
        Linear_RMSE_mean
        ...
    """

    # =====================================================
    # Long summary first
    # =====================================================

    summary_long = (
        summarize_performance(
            results
        )
    )


    if len(
        summary_long
    ) == 0:

        return pd.DataFrame()


    # =====================================================
    # ID columns
    # =====================================================

    id_columns = [
        "TrainSubjects",
        "Model",
    ]


    # =====================================================
    # Value columns
    # =====================================================

    value_columns = [
        c
        for c in summary_long.columns
        if c not in id_columns
    ]


    # =====================================================
    # Wide conversion
    # =====================================================

    wide = (
        summary_long
        .set_index(
            [
                "TrainSubjects",
                "Model",
            ]
        )[
            value_columns
        ]
        .unstack(
            "Model"
        )
    )


    # -----------------------------------------------------
    # Flatten columns
    # -----------------------------------------------------

    wide.columns = [
        f"{model}_{metric}"
        for metric, model
        in wide.columns
    ]


    wide = (
        wide
        .reset_index()
    )


    return wide