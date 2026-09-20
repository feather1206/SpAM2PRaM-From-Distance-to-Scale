# =========================================================
# Pipeline
# =========================================================

from config import PRAM_DIR, SPAM_DIR

from data import (
    read_pram_folder,
    read_spam_folder,
    average_spam,
)

from preprocessing import (
    matrix_to_pairs,
)


def load_dataset():
    """
    Load and preprocess datasets.

    Returns
    -------
    pram : DataFrame

    spam_pairs : DataFrame
    """

    pram = read_pram_folder(PRAM_DIR)

    spam_matrices = read_spam_folder(SPAM_DIR)

    avg_matrix = average_spam(spam_matrices)

    spam_pairs = matrix_to_pairs(avg_matrix)

    return pram, spam_pairs