# =========================================================
# pipeline.py
#
# Load individual-level PRaM and SpAM data.
#
# IMPORTANT:
# SpAM is NOT averaged here.
#
# Train/Test subjects are determined later in model.py.
# After splitting subjects:
#
# Train subjects
#     Mean SpAM
#         ->
#     Individual PRaM
#
# Test subjects
#     Mean SpAM
#         ->
#     Predicted PRaM
#
# =========================================================


from config import (
    PRAM_DIR,
    SPAM_DIR,
)

from data import (
    read_pram_folder,
    read_spam_folder,
)


# =========================================================
# Load Dataset
# =========================================================

def load_dataset():
    """
    Load PRaM and SpAM datasets.

    Returns
    -------
    pram : pandas.DataFrame

        Individual-level PRaM data.

        Columns
        -------
        Subject
        A
        B
        Resp


    spam : pandas.DataFrame

        Individual-level SpAM data.

        Columns
        -------
        Subject
        A
        B
        Distance


    Notes
    -----
    SpAM distances are intentionally NOT averaged here.

    Subject-level SpAM data are required because
    model.py first divides participants into:

        Train subjects
        Test subjects

    and calculates the mean SpAM distance separately
    within each group.
    """

    # =====================================================
    # Load PRaM
    # =====================================================

    print(
        "Loading PRaM data..."
    )

    pram = read_pram_folder(
        PRAM_DIR
    )


    # =====================================================
    # Load SpAM
    # =====================================================

    print(
        "Loading SpAM data..."
    )

    spam = read_spam_folder(
        SPAM_DIR
    )


    # =====================================================
    # Basic checks
    # =====================================================

    required_pram_columns = {
        "Subject",
        "A",
        "B",
        "Resp",
    }

    required_spam_columns = {
        "Subject",
        "A",
        "B",
        "Distance",
    }


    # -----------------------------------------------------
    # PRaM columns
    # -----------------------------------------------------

    missing_pram_columns = (
        required_pram_columns
        -
        set(
            pram.columns
        )
    )

    if missing_pram_columns:

        raise ValueError(
            "Missing PRaM columns: "
            f"{sorted(missing_pram_columns)}"
        )


    # -----------------------------------------------------
    # SpAM columns
    # -----------------------------------------------------

    missing_spam_columns = (
        required_spam_columns
        -
        set(
            spam.columns
        )
    )

    if missing_spam_columns:

        raise ValueError(
            "Missing SpAM columns: "
            f"{sorted(missing_spam_columns)}"
        )


    # =====================================================
    # Subject IDs -> int
    # =====================================================

    pram = pram.copy()

    spam = spam.copy()


    pram[
        "Subject"
    ] = (
        pram[
            "Subject"
        ]
        .astype(int)
    )


    spam[
        "Subject"
    ] = (
        spam[
            "Subject"
        ]
        .astype(int)
    )


    # =====================================================
    # Stimulus IDs -> int
    # =====================================================

    for column in [
        "A",
        "B",
    ]:

        pram[
            column
        ] = (
            pram[
                column
            ]
            .astype(int)
        )

        spam[
            column
        ] = (
            spam[
                column
            ]
            .astype(int)
        )


    # =====================================================
    # Response / Distance -> numeric
    # =====================================================

    pram[
        "Resp"
    ] = (
        pram[
            "Resp"
        ]
        .astype(float)
    )


    spam[
        "Distance"
    ] = (
        spam[
            "Distance"
        ]
        .astype(float)
    )


    # =====================================================
    # Remove missing values
    # =====================================================

    pram = (
        pram
        .dropna(
            subset=[
                "Subject",
                "A",
                "B",
                "Resp",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    spam = (
        spam
        .dropna(
            subset=[
                "Subject",
                "A",
                "B",
                "Distance",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # Standardize pair order
    #
    # Make sure:
    #
    # A < B
    #
    # =====================================================

    pram_a = (
        pram[
            [
                "A",
                "B",
            ]
        ]
        .min(
            axis=1
        )
    )

    pram_b = (
        pram[
            [
                "A",
                "B",
            ]
        ]
        .max(
            axis=1
        )
    )

    pram[
        "A"
    ] = pram_a

    pram[
        "B"
    ] = pram_b


    spam_a = (
        spam[
            [
                "A",
                "B",
            ]
        ]
        .min(
            axis=1
        )
    )

    spam_b = (
        spam[
            [
                "A",
                "B",
            ]
        ]
        .max(
            axis=1
        )
    )

    spam[
        "A"
    ] = spam_a

    spam[
        "B"
    ] = spam_b


    # =====================================================
    # Sort
    # =====================================================

    pram = (
        pram
        .sort_values(
            [
                "Subject",
                "A",
                "B",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    spam = (
        spam
        .sort_values(
            [
                "Subject",
                "A",
                "B",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # Subjects existing in both datasets
    # =====================================================

    pram_subjects = set(
        pram[
            "Subject"
        ].unique()
    )

    spam_subjects = set(
        spam[
            "Subject"
        ].unique()
    )

    common_subjects = sorted(
        pram_subjects
        &
        spam_subjects
    )


    # =====================================================
    # Keep common subjects only
    # =====================================================

    pram = (
        pram[
            pram[
                "Subject"
            ].isin(
                common_subjects
            )
        ]
        .reset_index(
            drop=True
        )
    )


    spam = (
        spam[
            spam[
                "Subject"
            ].isin(
                common_subjects
            )
        ]
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # Display information
    # =====================================================

    print()

    print(
        "=" * 60
    )

    print(
        "Dataset Information"
    )

    print(
        "=" * 60
    )


    print(
        f"PRaM rows     : "
        f"{len(pram)}"
    )

    print(
        f"SpAM rows     : "
        f"{len(spam)}"
    )

    print(
        f"Subjects      : "
        f"{len(common_subjects)}"
    )


    print(
        f"PRaM pairs    : "
        f"{pram[['A', 'B']].drop_duplicates().shape[0]}"
    )

    print(
        f"SpAM pairs    : "
        f"{spam[['A', 'B']].drop_duplicates().shape[0]}"
    )


    # =====================================================
    # Return
    # =====================================================

    return (
        pram,
        spam,
    )