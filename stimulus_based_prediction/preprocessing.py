import pandas as pd
import numpy as np

def matrix_to_pairs(matrix):
    """
    Convert distance matrix to pair dataframe.
    """

    n_items = matrix.shape[0]

    rows = []

    for i in range(n_items):

        for j in range(i + 1, n_items):

            rows.append(
                {
                    "A": i + 1,
                    "B": j + 1,
                    "Distance": matrix[i, j]
                }
            )

    return pd.DataFrame(rows)

# =========================================================
# Train/Test Split
# =========================================================

def make_train_test(pair_df, train_items):
    """
    Train:
        Both products belong to train_items

    Test:
        Others
    """

    train_items = set(train_items)

    train_mask = (
        pair_df["A"].isin(train_items)
        &
        pair_df["B"].isin(train_items)
    )

    train = pair_df.loc[train_mask].copy()

    test = pair_df.loc[~train_mask].copy()

    return train, test

def get_all_items(pair_df):
    """
    Get all item numbers.
    """

    items = sorted(
        set(pair_df["A"]).union(pair_df["B"])
    )

    return np.array(items)