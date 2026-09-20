import pandas as pd
import numpy as np
import glob
import os
import re

from scipy.stats import pearsonr, spearmanr
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform

from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


# ==========================================
# 1. Settings
# ==========================================

# Predicted PRaM
PREDICT_FILE = "results/PRaM_prediction.csv"

# Measured PRaM
PRAM_DIR = "PRaM_data"

# Stimulus images
IMAGE_DIR = "image"

# Output
OUTPUT_DIR = "average_results"

ITEM_COUNT = 16

LIKERT_MAX = 7

IS_SIMILARITY = True

FILENAME_IS_1_BASED = True

N_PERMUTATIONS = 10000

RANDOM_STATE = 42


# ==========================================
# 2. Utility
# ==========================================

def rating_to_distance(rating):
    """
    Convert PRaM similarity rating to dissimilarity.

    7 -> 1
    6 -> 2
    ...
    1 -> 7
    """

    if IS_SIMILARITY:

        return (
            LIKERT_MAX
            -
            rating
            +
            1
        )

    return rating


# ==========================================
# 3. Load measured PRaM
# ==========================================

def load_pram_matrix(
    filepath,
    n_items,
):
    """
    Create PRaM dissimilarity matrix directly
    from the original PRaM CSV.

    Required columns:
        images_ocha_A
        key_resp_2.keys
    """

    try:

        df = pd.read_csv(
            filepath
        )

        matrix = np.zeros(
            (
                n_items,
                n_items,
            ),
            dtype=float,
        )

        for _, row in df.iterrows():

            img_path = row[
                "images_ocha_A"
            ]

            rating = row[
                "key_resp_2.keys"
            ]

            # ----------------------------------
            # Missing value
            # ----------------------------------

            if (
                not isinstance(
                    img_path,
                    str
                )
                or
                pd.isna(
                    rating
                )
            ):

                continue

            # ----------------------------------
            # Extract stimulus pair IDs
            #
            # Example:
            # 1_2 -> 1, 2
            # ----------------------------------

            ids = re.findall(
                r"(\d+)_(\d+)",
                img_path
            )

            if not ids:

                continue

            id_a, id_b = map(
                int,
                ids[0]
            )

            if FILENAME_IS_1_BASED:

                idx_a = id_a - 1
                idx_b = id_b - 1

            else:

                idx_a = id_a
                idx_b = id_b

            # ----------------------------------
            # Range check
            # ----------------------------------

            if not (
                0 <= idx_a < n_items
                and
                0 <= idx_b < n_items
            ):

                continue

            # ----------------------------------
            # Similarity -> dissimilarity
            # ----------------------------------

            distance = rating_to_distance(
                rating
            )

            matrix[
                idx_a,
                idx_b
            ] = distance

            matrix[
                idx_b,
                idx_a
            ] = distance

        np.fill_diagonal(
            matrix,
            0
        )

        return matrix

    except Exception:

        return None


# ==========================================
# 4. Average measured PRaM
# ==========================================

def load_average_measured_pram(
    pram_dir,
    n_items,
):
    """
    Load all PRaM participant files and
    calculate the participant-mean
    dissimilarity matrix.
    """

    pram_files = glob.glob(
        os.path.join(
            pram_dir,
            "*.csv"
        )
    )

    pram_matrices = []

    for filepath in pram_files:

        matrix = load_pram_matrix(
            filepath,
            n_items,
        )

        if matrix is None:

            continue

        # ----------------------------------
        # Check missing pairs
        # ----------------------------------

        upper = matrix[
            np.triu_indices(
                n_items,
                k=1
            )
        ]

        if np.any(
            upper == 0
        ):

            continue

        pram_matrices.append(
            matrix
        )

    if len(
        pram_matrices
    ) == 0:

        raise ValueError(
            "No valid PRaM data were found "
            "in PRaM_data."
        )

    measured_matrix = np.mean(
        pram_matrices,
        axis=0
    )

    np.fill_diagonal(
        measured_matrix,
        0
    )

    return measured_matrix


# ==========================================
# 5. Load predicted PRaM
# ==========================================

def load_prediction_matrix(
    filepath,
    n_items,
):
    """
    Load predicted PRaM from:

        results/PRaM_prediction.csv

    Required columns:
        A
        B
        PRaM_predicted
    """

    df = pd.read_csv(
        filepath
    )

    required_columns = [
        "A",
        "B",
        "PRaM_predicted",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in prediction file: "
            +
            ", ".join(
                missing_columns
            )
        )

    pred_matrix = np.zeros(
        (
            n_items,
            n_items,
        ),
        dtype=float,
    )

    for _, row in df.iterrows():

        if pd.isna(
            row["PRaM_predicted"]
        ):

            continue

        a = int(
            row["A"]
        ) - 1

        b = int(
            row["B"]
        ) - 1

        pred_rating = row[
            "PRaM_predicted"
        ]

        pred_distance = rating_to_distance(
            pred_rating
        )

        pred_matrix[
            a,
            b
        ] = pred_distance

        pred_matrix[
            b,
            a
        ] = pred_distance

    np.fill_diagonal(
        pred_matrix,
        0
    )

    # --------------------------------------
    # Check all stimulus pairs exist
    # --------------------------------------

    upper = pred_matrix[
        np.triu_indices(
            n_items,
            k=1
        )
    ]

    if np.any(
        upper == 0
    ):

        raise ValueError(
            "PRaM_prediction.csv does not "
            "contain predictions for all "
            "stimulus pairs."
        )

    return pred_matrix


# ==========================================
# 6. Plot
# ==========================================

def plot_with_images(
    ax,
    positions,
    title,
):

    for i in range(
        len(positions)
    ):

        x, y = positions[i]

        img_path = os.path.join(
            IMAGE_DIR,
            f"{i + 1}.png"
        )

        if not os.path.exists(
            img_path
        ):

            continue

        img = mpimg.imread(
            img_path
        )

        imagebox = OffsetImage(
            img,
            zoom=0.2
        )

        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=False
        )

        ax.add_artist(
            ab
        )

    ax.set_title(
        title
    )

    ax.set_xlim(
        positions[:, 0].min()
        -
        0.15,

        positions[:, 0].max()
        +
        0.15
    )

    ax.set_ylim(
        positions[:, 1].min()
        -
        0.15,

        positions[:, 1].max()
        +
        0.15
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )


# ==========================================
# 7. Save original matrices
# ==========================================

def save_average_matrices(
    pred_matrix,
    measured_matrix,
    output_dir=OUTPUT_DIR,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    labels = [
        f"Item_{i + 1}"
        for i in range(
            ITEM_COUNT
        )
    ]

    # --------------------------------------
    # Dissimilarity matrices
    # --------------------------------------

    pd.DataFrame(
        pred_matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "prediction_distance_matrix.csv",
        ),
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        measured_matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "measured_distance_matrix.csv",
        ),
        encoding="utf-8-sig",
    )

    # --------------------------------------
    # Likert matrices
    # --------------------------------------

    if IS_SIMILARITY:

        pred_likert = (
            LIKERT_MAX
            -
            pred_matrix
            +
            1
        )

        measured_likert = (
            LIKERT_MAX
            -
            measured_matrix
            +
            1
        )

        np.fill_diagonal(
            pred_likert,
            LIKERT_MAX
        )

        np.fill_diagonal(
            measured_likert,
            LIKERT_MAX
        )

        pd.DataFrame(
            pred_likert,
            index=labels,
            columns=labels,
        ).to_csv(
            os.path.join(
                output_dir,
                "prediction_likert_matrix.csv",
            ),
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            measured_likert,
            index=labels,
            columns=labels,
        ).to_csv(
            os.path.join(
                output_dir,
                "measured_likert_matrix.csv",
            ),
            encoding="utf-8-sig",
        )


# ==========================================
# 8. Analysis
# ==========================================

def compute_metrics(
    pred_matrix,
    measured_matrix,
    n_permutations=N_PERMUTATIONS,
    random_state=RANDOM_STATE,
):

    # --------------------------------------
    # Non-metric MDS : Prediction
    # --------------------------------------

    mds_pred = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=50,
        max_iter=3000,
        normalized_stress="auto",
    )

    pos_pred = mds_pred.fit_transform(
        pred_matrix
    )

    stress_pred = mds_pred.stress_


    # --------------------------------------
    # Non-metric MDS : Measured
    # --------------------------------------

    mds_measured = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=50,
        max_iter=3000,
        normalized_stress="auto",
    )

    pos_measured = (
        mds_measured.fit_transform(
            measured_matrix
        )
    )

    stress_measured = (
        mds_measured.stress_
    )


    # --------------------------------------
    # MDS distance vectors
    # --------------------------------------

    v_pred = pdist(
        pos_pred,
        metric="euclidean",
    )

    v_measured = pdist(
        pos_measured,
        metric="euclidean",
    )


    # --------------------------------------
    # Pearson
    # --------------------------------------

    r, r_p = pearsonr(
        v_pred,
        v_measured,
    )


    # --------------------------------------
    # Spearman
    # --------------------------------------

    rho, rho_p = spearmanr(
        v_pred,
        v_measured,
    )


    # --------------------------------------
    # Procrustes
    # --------------------------------------

    (
        aligned_pred,
        aligned_measured,
        disparity,
    ) = procrustes(
        pos_pred,
        pos_measured,
    )


    # ======================================
    # Permutation test
    #
    # Randomize stimulus correspondence.
    # Smaller disparity = stronger match.
    # ======================================

    rng = np.random.default_rng(
        random_state
    )

    n_items = pos_pred.shape[0]

    perm_disparities = np.empty(
        n_permutations,
        dtype=float,
    )

    for i in range(
        n_permutations
    ):

        permutation = rng.permutation(
            n_items
        )

        permuted_pred = pos_pred[
            permutation
        ]

        _, _, perm_disparity = procrustes(
            permuted_pred,
            pos_measured,
        )

        perm_disparities[
            i
        ] = perm_disparity


    # --------------------------------------
    # Permutation p-value
    # --------------------------------------

    permutation_p = (
        np.sum(
            perm_disparities
            <=
            disparity
        )
        +
        1
    ) / (
        n_permutations
        +
        1
    )


    # ======================================
    # Print analysis results ONLY
    # ======================================

    print(
        f"Pearson r : "
        f"{r:.3f} "
        f"(p = {r_p:.6g})"
    )

    print(
        f"Spearman rho : "
        f"{rho:.3f} "
        f"(p = {rho_p:.6g})"
    )

    print(
        f"Prediction Stress : "
        f"{stress_pred:.5f}"
    )

    print(
        f"Measured Stress : "
        f"{stress_measured:.5f}"
    )

    print(
        f"Procrustes disparity : "
        f"{disparity:.5f}"
    )

    print(
        f"Permutation p : "
        f"{permutation_p:.6f}"
    )


    return (
        aligned_pred,
        aligned_measured,
        pos_pred,
        pos_measured,
        r,
        rho,
        stress_pred,
        stress_measured,
        disparity,
        permutation_p,
        perm_disparities,
    )


# ==========================================
# 9. Save MDS results
# ==========================================

def save_mds_results(
    pos_pred,
    pos_measured,
    output_dir=OUTPUT_DIR,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    labels = [
        f"Item_{i + 1}"
        for i in range(
            ITEM_COUNT
        )
    ]

    # --------------------------------------
    # Coordinates
    # --------------------------------------

    coord_pred = pd.DataFrame(
        {
            "Item": labels,
            "X": pos_pred[:, 0],
            "Y": pos_pred[:, 1],
        }
    )

    coord_measured = pd.DataFrame(
        {
            "Item": labels,
            "X": pos_measured[:, 0],
            "Y": pos_measured[:, 1],
        }
    )

    coord_pred.to_csv(
        os.path.join(
            output_dir,
            "mds_prediction_coordinates.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    coord_measured.to_csv(
        os.path.join(
            output_dir,
            "mds_measured_coordinates.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------
    # MDS Euclidean distance matrices
    # --------------------------------------

    dist_pred = squareform(
        pdist(
            pos_pred
        )
    )

    dist_measured = squareform(
        pdist(
            pos_measured
        )
    )

    pd.DataFrame(
        dist_pred,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "mds_prediction_distance.csv",
        ),
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        dist_measured,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "mds_measured_distance.csv",
        ),
        encoding="utf-8-sig",
    )


# ==========================================
# 10. Main
# ==========================================

# ------------------------------------------
# Predicted PRaM
#
# Source:
# results/PRaM_prediction.csv
# ------------------------------------------

prediction_matrix = (
    load_prediction_matrix(
        PREDICT_FILE,
        ITEM_COUNT,
    )
)


# ------------------------------------------
# Measured mean PRaM
#
# Source:
# PRaM_data/*.csv
#
# columns:
# images_ocha_A
# key_resp_2.keys
# ------------------------------------------

measured_matrix = (
    load_average_measured_pram(
        PRAM_DIR,
        ITEM_COUNT,
    )
)


# ------------------------------------------
# Save original matrices
# ------------------------------------------

save_average_matrices(
    prediction_matrix,
    measured_matrix,
    output_dir=OUTPUT_DIR,
)


# ------------------------------------------
# Analysis
# ------------------------------------------

(
    aligned_pred,
    aligned_measured,
    raw_pred,
    raw_measured,
    pearson_r,
    spearman_rho,
    stress_pred,
    stress_measured,
    disparity,
    permutation_p,
    perm_disparities,
) = compute_metrics(
    prediction_matrix,
    measured_matrix,
    n_permutations=N_PERMUTATIONS,
    random_state=RANDOM_STATE,
)


# ------------------------------------------
# Save MDS results
# ------------------------------------------

save_mds_results(
    raw_pred,
    raw_measured,
    output_dir=OUTPUT_DIR,
)


# ==========================================
# 11. Plot
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        16,
        8,
    ),
)

plot_with_images(
    axes[0],
    aligned_pred,
    "Predicted PRaM",
)

plot_with_images(
    axes[1],
    aligned_measured,
    "Measured Mean PRaM",
)

plt.suptitle(
    "Predicted vs Measured PRaM"
)

plt.tight_layout()

plt.show()

import pandas as pd
import numpy as np
import glob
import os

from scipy.stats import pearsonr, spearmanr
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform

from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


# ==========================================
# 1. Settings
# ==========================================

# Original SpAM data
SPAM_DIR = "SpAM_data"

# Predicted PRaM
PREDICT_FILE = "results/PRaM_prediction.csv"

# Stimulus images
IMAGE_DIR = "image"

# Output
OUTPUT_DIR = "spam_vs_prediction_results"

ITEM_COUNT = 16

LIKERT_MAX = 7

IS_SIMILARITY = True

N_PERMUTATIONS = 10000

RANDOM_STATE = 42


# ==========================================
# 2. Utility
# ==========================================

def rating_to_distance(rating):
    """
    Convert predicted PRaM similarity rating
    to dissimilarity.

    7 -> 1
    6 -> 2
    ...
    1 -> 7
    """

    if IS_SIMILARITY:

        return (
            LIKERT_MAX
            -
            rating
            +
            1
        )

    return rating


# ==========================================
# 3. Load individual SpAM matrix
# ==========================================

def load_spam_matrix(filepath):
    """
    Load original SpAM distance matrix.

    Same format as the previous
    SpAM / PRaM analysis.
    """

    try:

        df = pd.read_csv(
            filepath,
            skiprows=22,
            header=None,
        )

        matrix = df.values.astype(
            float
        )

        return matrix

    except Exception:

        return None


# ==========================================
# 4. Calculate mean SpAM matrix
# ==========================================

def load_average_spam(
    spam_dir,
    n_items,
):
    """
    Load all individual SpAM matrices
    and calculate participant mean.
    """

    spam_files = glob.glob(
        os.path.join(
            spam_dir,
            "*.csv"
        )
    )

    spam_matrices = []

    for filepath in spam_files:

        matrix = load_spam_matrix(
            filepath
        )

        if matrix is None:

            continue

        if matrix.shape != (
            n_items,
            n_items,
        ):

            continue

        if np.isnan(
            matrix
        ).any():

            continue

        spam_matrices.append(
            matrix
        )

    if len(
        spam_matrices
    ) == 0:

        raise ValueError(
            "No valid SpAM matrices "
            "were found."
        )

    avg_spam = np.mean(
        spam_matrices,
        axis=0,
    )

    np.fill_diagonal(
        avg_spam,
        0
    )

    return avg_spam


# ==========================================
# 5. Load predicted PRaM matrix
# ==========================================

def load_prediction_matrix(
    filepath,
    n_items,
):
    """
    Load predicted PRaM from:

        results/PRaM_prediction.csv

    Required:
        A
        B
        PRaM_predicted
    """

    df = pd.read_csv(
        filepath
    )

    required_columns = [
        "A",
        "B",
        "PRaM_predicted",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            +
            ", ".join(
                missing_columns
            )
        )

    pred_matrix = np.zeros(
        (
            n_items,
            n_items,
        ),
        dtype=float,
    )

    for _, row in df.iterrows():

        if pd.isna(
            row["PRaM_predicted"]
        ):

            continue

        a = int(
            row["A"]
        ) - 1

        b = int(
            row["B"]
        ) - 1

        predicted_rating = row[
            "PRaM_predicted"
        ]

        predicted_distance = (
            rating_to_distance(
                predicted_rating
            )
        )

        pred_matrix[
            a,
            b
        ] = predicted_distance

        pred_matrix[
            b,
            a
        ] = predicted_distance

    np.fill_diagonal(
        pred_matrix,
        0
    )

    # Check all 120 pairs
    upper = pred_matrix[
        np.triu_indices(
            n_items,
            k=1,
        )
    ]

    if np.any(
        upper == 0
    ):

        raise ValueError(
            "Prediction data are missing "
            "one or more stimulus pairs."
        )

    return pred_matrix


# ==========================================
# 6. Plot
# ==========================================

def plot_with_images(
    ax,
    positions,
    title,
):

    for i in range(
        len(positions)
    ):

        x, y = positions[i]

        img_path = os.path.join(
            IMAGE_DIR,
            f"{i + 1}.png"
        )

        if not os.path.exists(
            img_path
        ):

            continue

        img = mpimg.imread(
            img_path
        )

        imagebox = OffsetImage(
            img,
            zoom=0.2,
        )

        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=False,
        )

        ax.add_artist(
            ab
        )

    ax.set_title(
        title
    )

    ax.set_xlim(
        positions[:, 0].min()
        -
        0.15,
        positions[:, 0].max()
        +
        0.15,
    )

    ax.set_ylim(
        positions[:, 1].min()
        -
        0.15,
        positions[:, 1].max()
        +
        0.15,
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )


# ==========================================
# 7. Save original matrices
# ==========================================

def save_original_matrices(
    spam_matrix,
    predicted_matrix,
    output_dir=OUTPUT_DIR,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    labels = [
        f"Item_{i + 1}"
        for i in range(
            ITEM_COUNT
        )
    ]

    # SpAM distance matrix
    pd.DataFrame(
        spam_matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "average_spam_distance.csv",
        ),
        encoding="utf-8-sig",
    )

    # Predicted PRaM distance matrix
    pd.DataFrame(
        predicted_matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "predicted_pram_distance.csv",
        ),
        encoding="utf-8-sig",
    )

    # Predicted PRaM Likert matrix
    predicted_likert = (
        LIKERT_MAX
        -
        predicted_matrix
        +
        1
    )

    np.fill_diagonal(
        predicted_likert,
        LIKERT_MAX,
    )

    pd.DataFrame(
        predicted_likert,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "predicted_pram_likert.csv",
        ),
        encoding="utf-8-sig",
    )


# ==========================================
# 8. Analysis
# ==========================================

def compute_metrics(
    spam_matrix,
    predicted_matrix,
    n_permutations=N_PERMUTATIONS,
    random_state=RANDOM_STATE,
):

    # --------------------------------------
    # Non-metric MDS : SpAM
    # --------------------------------------

    mds_spam = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=50,
        max_iter=3000,
        normalized_stress="auto",
    )

    pos_spam = mds_spam.fit_transform(
        spam_matrix
    )

    stress_spam = (
        mds_spam.stress_
    )


    # --------------------------------------
    # Non-metric MDS : Predicted PRaM
    # --------------------------------------

    mds_pred = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=50,
        max_iter=3000,
        normalized_stress="auto",
    )

    pos_pred = mds_pred.fit_transform(
        predicted_matrix
    )

    stress_pred = (
        mds_pred.stress_
    )


    # --------------------------------------
    # MDS distance vectors
    # --------------------------------------

    v_spam = pdist(
        pos_spam,
        metric="euclidean",
    )

    v_pred = pdist(
        pos_pred,
        metric="euclidean",
    )


    # --------------------------------------
    # Pearson
    # --------------------------------------

    r, r_p = pearsonr(
        v_spam,
        v_pred,
    )


    # --------------------------------------
    # Spearman
    # --------------------------------------

    rho, rho_p = spearmanr(
        v_spam,
        v_pred,
    )


    # --------------------------------------
    # Procrustes
    # --------------------------------------

    (
        aligned_spam,
        aligned_pred,
        disparity,
    ) = procrustes(
        pos_spam,
        pos_pred,
    )


    # ======================================
    # Permutation test
    # ======================================

    rng = np.random.default_rng(
        random_state
    )

    n_items = pos_spam.shape[0]

    perm_disparities = np.empty(
        n_permutations,
        dtype=float,
    )

    for i in range(
        n_permutations
    ):

        permutation = rng.permutation(
            n_items
        )

        permuted_pred = pos_pred[
            permutation
        ]

        _, _, perm_disparity = (
            procrustes(
                pos_spam,
                permuted_pred,
            )
        )

        perm_disparities[
            i
        ] = perm_disparity


    # --------------------------------------
    # Permutation p-value
    # --------------------------------------

    permutation_p = (
        np.sum(
            perm_disparities
            <=
            disparity
        )
        +
        1
    ) / (
        n_permutations
        +
        1
    )


    # ======================================
    # Print analysis results ONLY
    # ======================================

    print(
        f"Pearson r : "
        f"{r:.6f} "
        f"(p = {r_p:.6g})"
    )

    print(
        f"Spearman rho : "
        f"{rho:.6f} "
        f"(p = {rho_p:.6g})"
    )

    print(
        f"SpAM Stress : "
        f"{stress_spam:.10f}"
    )

    print(
        f"Predicted PRaM Stress : "
        f"{stress_pred:.10f}"
    )

    print(
        f"Procrustes disparity : "
        f"{disparity:.10f}"
    )

    print(
        f"Permutation p : "
        f"{permutation_p:.6f}"
    )


    return (
        aligned_spam,
        aligned_pred,
        pos_spam,
        pos_pred,
        r,
        rho,
        stress_spam,
        stress_pred,
        disparity,
        permutation_p,
        perm_disparities,
    )


# ==========================================
# 9. Save MDS results
# ==========================================

def save_mds_results(
    pos_spam,
    pos_pred,
    output_dir=OUTPUT_DIR,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    labels = [
        f"Item_{i + 1}"
        for i in range(
            ITEM_COUNT
        )
    ]

    # --------------------------------------
    # Coordinates
    # --------------------------------------

    coord_spam = pd.DataFrame(
        {
            "Item": labels,
            "X": pos_spam[:, 0],
            "Y": pos_spam[:, 1],
        }
    )

    coord_pred = pd.DataFrame(
        {
            "Item": labels,
            "X": pos_pred[:, 0],
            "Y": pos_pred[:, 1],
        }
    )

    coord_spam.to_csv(
        os.path.join(
            output_dir,
            "mds_spam_coordinates.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    coord_pred.to_csv(
        os.path.join(
            output_dir,
            "mds_predicted_pram_coordinates.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------
    # MDS Euclidean distance matrices
    # --------------------------------------

    dist_spam = squareform(
        pdist(
            pos_spam
        )
    )

    dist_pred = squareform(
        pdist(
            pos_pred
        )
    )

    pd.DataFrame(
        dist_spam,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "mds_spam_distance.csv",
        ),
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        dist_pred,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            output_dir,
            "mds_predicted_pram_distance.csv",
        ),
        encoding="utf-8-sig",
    )


# ==========================================
# 10. Main
# ==========================================

# ------------------------------------------
# Mean SpAM
#
# Source:
# SpAM_data/*.csv
# ------------------------------------------

spam_matrix = (
    load_average_spam(
        SPAM_DIR,
        ITEM_COUNT,
    )
)


# ------------------------------------------
# Predicted PRaM
#
# Source:
# results/PRaM_prediction.csv
# ------------------------------------------

predicted_matrix = (
    load_prediction_matrix(
        PREDICT_FILE,
        ITEM_COUNT,
    )
)


# ------------------------------------------
# Save original matrices
# ------------------------------------------

save_original_matrices(
    spam_matrix,
    predicted_matrix,
    output_dir=OUTPUT_DIR,
)


# ------------------------------------------
# Analysis
# ------------------------------------------

(
    aligned_spam,
    aligned_pred,
    raw_spam,
    raw_pred,
    pearson_r,
    spearman_rho,
    stress_spam,
    stress_pred,
    disparity,
    permutation_p,
    perm_disparities,
) = compute_metrics(
    spam_matrix,
    predicted_matrix,
    n_permutations=N_PERMUTATIONS,
    random_state=RANDOM_STATE,
)


# ------------------------------------------
# Save MDS results
# ------------------------------------------

save_mds_results(
    raw_spam,
    raw_pred,
    output_dir=OUTPUT_DIR,
)


# ==========================================
# 11. Plot
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        16,
        8,
    ),
)

plot_with_images(
    axes[0],
    aligned_spam,
    "Mean SpAM",
)

plot_with_images(
    axes[1],
    aligned_pred,
    "Predicted PRaM",
)

plt.suptitle(
    "Mean SpAM vs Predicted PRaM"
)

plt.tight_layout()

plt.show()
