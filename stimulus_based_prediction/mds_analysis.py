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
# 1. 設定
# ==========================================

PREDICT_FILE = "results/PRaM_completed.csv"
PRAM_DIR = "PRaM_data"
IMAGE_DIR = "image"

ITEM_COUNT = 16
LIKERT_MAX = 7
IS_SIMILARITY = True
FILENAME_IS_1_BASED = True

VAR_THRESHOLD = 0.5

# ------------------------------------------
# Procrustes permutation test
# ------------------------------------------

N_PERMUTATIONS = 10000
PERMUTATION_RANDOM_STATE = 42


# ==========================================
# 2. 関数
# ==========================================

def load_prediction_matrix(
    filepath,
    n_items,
):
    """
    PRaM_completed.csv

    A, B, PRaM, DataType

    ↓

    距離行列へ変換
    """

    df = pd.read_csv(
        filepath
    )

    matrix = np.zeros(
        (
            n_items,
            n_items,
        )
    )

    for _, row in df.iterrows():

        a = int(
            row["A"]
        ) - 1

        b = int(
            row["B"]
        ) - 1

        rating = row[
            "PRaM"
        ]

        if IS_SIMILARITY:

            dist = (
                LIKERT_MAX
                -
                rating
                +
                1
            )

        else:

            dist = rating

        matrix[
            a,
            b
        ] = dist

        matrix[
            b,
            a
        ] = dist

    np.fill_diagonal(
        matrix,
        0
    )

    return matrix


# ==========================================

def load_pram_matrix(
    filepath,
    n_items,
):
    """
    PRaM形式のCSVから距離行列を生成
    """

    try:

        df = pd.read_csv(
            filepath
        )

        matrix = np.zeros(
            (
                n_items,
                n_items,
            )
        )

        for _, row in df.iterrows():

            img_path = row[
                "images_ocha_A"
            ]

            rating = row[
                "key_resp_2.keys"
            ]

            if (
                not isinstance(
                    img_path,
                    str
                )
                or
                pd.isna(rating)
            ):

                continue

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

            if not (
                0 <= idx_a < n_items
                and
                0 <= idx_b < n_items
            ):

                continue

            if IS_SIMILARITY:

                dist = (
                    LIKERT_MAX
                    -
                    rating
                    +
                    1
                )

            else:

                dist = rating

            matrix[
                idx_a,
                idx_b
            ] = dist

            matrix[
                idx_b,
                idx_a
            ] = dist

        return matrix

    except Exception as e:

        print(
            f"[ERROR] {filepath}"
        )

        print(e)

        return None


# ==========================================

def get_upper_triangle(
    matrix,
):

    return matrix[
        np.triu_indices(
            matrix.shape[0],
            k=1
        )
    ]


# ==========================================

def is_bad_response(
    vec,
):

    vec = vec[
        ~np.isnan(vec)
    ]

    if len(vec) == 0:

        return True

    return (
        np.var(vec)
        <
        VAR_THRESHOLD
    )


# ==========================================
# Plot
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
# Procrustes permutation test
# ==========================================

def procrustes_permutation_test(
    pos_pred,
    pos_pram,
    n_permutations=10000,
    random_state=42,
):
    """
    Procrustes disparity に対する permutation test。

    帰無仮説:
        Prediction空間とMeasured PRaM空間の
        刺激対応はランダムである。

    PRaM側の刺激ラベルをランダムに並べ替え、
    各並べ替えについてProcrustes disparityを算出する。

    disparityは小さいほど一致度が高いため、
    観測disparity以下となる割合をp値とする。
    """

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------
    # Observed disparity
    # --------------------------------------

    _, _, observed_disparity = (
        procrustes(
            pos_pred,
            pos_pram
        )
    )

    # --------------------------------------
    # Null distribution
    # --------------------------------------

    null_disparities = np.empty(
        n_permutations,
        dtype=float
    )

    n_items = pos_pram.shape[0]

    for i in range(
        n_permutations
    ):

        permutation = rng.permutation(
            n_items
        )

        permuted_pram = pos_pram[
            permutation
        ]

        _, _, perm_disparity = (
            procrustes(
                pos_pred,
                permuted_pram
            )
        )

        null_disparities[
            i
        ] = perm_disparity

    # --------------------------------------
    # p-value
    # --------------------------------------

    count = np.sum(
        null_disparities
        <=
        observed_disparity
    )

    p_value = (
        count + 1
    ) / (
        n_permutations + 1
    )

    return (
        observed_disparity,
        null_disparities,
        p_value
    )


# ==========================================
# Permutation distribution plot
# ==========================================

def plot_permutation_distribution(
    null_disparities,
    observed_disparity,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        null_disparities,
        bins=40
    )

    plt.axvline(
        observed_disparity,
        linestyle="--",
        linewidth=2,
        label=(
            f"Observed = "
            f"{observed_disparity:.4f}"
        )
    )

    plt.xlabel(
        "Procrustes disparity"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Procrustes Permutation Test\n"
        "Predicted vs Measured PRaM"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ==========================================
# 平均保存
# ==========================================

def save_average_matrices(
    pred_matrix,
    pram_list,
    output_dir="average_results",
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    avg_pram_distance = np.mean(
        pram_list,
        axis=0
    )

    avg_pred = pred_matrix.copy()

    if IS_SIMILARITY:

        avg_pram_likert = (
            LIKERT_MAX
            -
            avg_pram_distance
            +
            1
        )

        np.fill_diagonal(
            avg_pram_likert,
            LIKERT_MAX
        )

    else:

        avg_pram_likert = (
            avg_pram_distance.copy()
        )

    np.fill_diagonal(
        avg_pred,
        0
    )

    labels = [
        f"Item_{i + 1}"
        for i in range(
            ITEM_COUNT
        )
    ]

    df_pred = pd.DataFrame(
        avg_pred,
        index=labels,
        columns=labels
    )

    df_pram = pd.DataFrame(
        avg_pram_likert,
        index=labels,
        columns=labels
    )

    df_pred.to_csv(
        os.path.join(
            output_dir,
            "average_prediction_distance.csv"
        ),
        encoding="utf-8-sig"
    )

    df_pram.to_csv(
        os.path.join(
            output_dir,
            "average_pram_likert.csv"
        ),
        encoding="utf-8-sig"
    )


# ==========================================
# 3. データ読み込み
# ==========================================

prediction_matrix = (
    load_prediction_matrix(
        PREDICT_FILE,
        ITEM_COUNT
    )
)

pram_matrices = []

bad_indices = []

print(
    "Starting Data Loading..."
)

pram_files = glob.glob(
    os.path.join(
        PRAM_DIR,
        "*.csv"
    )
)

for pram_path in pram_files:

    filename = os.path.basename(
        pram_path
    )

    m_pram = load_pram_matrix(
        pram_path,
        ITEM_COUNT
    )

    if m_pram is None:

        continue

    has_pram_missing = np.any(
        get_upper_triangle(
            m_pram
        )
        ==
        0
    )

    if has_pram_missing:

        print(
            f"[WARNING] {filename}"
        )

    vec_pram = get_upper_triangle(
        m_pram
    )

    if is_bad_response(
        vec_pram
    ):

        print(
            f"[REMOVED] {filename}"
        )

        bad_indices.append(
            len(
                pram_matrices
            )
        )

    pram_matrices.append(
        m_pram
    )


# ==========================================
# Filtering
# ==========================================

mask = np.ones(
    len(
        pram_matrices
    ),
    dtype=bool
)

mask[
    bad_indices
] = False


# ==========================================
# 4. 分析
# ==========================================

def compute_metrics(
    pred_matrix,
    pram_list,
    label,
):

    avg_pred = pred_matrix

    avg_pram = np.mean(
        pram_list,
        axis=0
    )

    # ======================================
    # Nonmetric MDS: Prediction
    # ======================================

    mds_pred = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=42,
        n_init=50,
        max_iter=3000,
        normalized_stress="auto"
    )

    pos_pred = mds_pred.fit_transform(
        avg_pred
    )

    stress_pred = (
        mds_pred.stress_
    )

    # ======================================
    # Nonmetric MDS: Measured PRaM
    # ======================================

    mds_pram = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=42,
        n_init=50,
        max_iter=3000,
        normalized_stress="auto"
    )

    pos_pram = mds_pram.fit_transform(
        avg_pram
    )

    stress_pram = (
        mds_pram.stress_
    )

    # ======================================
    # Hout et al. style distance comparison
    # ======================================

    v_pred = pdist(
        pos_pred,
        metric="euclidean"
    )

    v_pram = pdist(
        pos_pram,
        metric="euclidean"
    )

    r, r_p = pearsonr(
        v_pred,
        v_pram
    )

    rho, rho_p = spearmanr(
        v_pred,
        v_pram
    )

    # ======================================
    # Procrustes
    # ======================================

    (
        aligned_pred,
        aligned_pram,
        disparity
    ) = procrustes(
        pos_pred,
        pos_pram
    )

    # ======================================
    # Procrustes permutation test
    # ======================================

    (
        observed_disparity,
        null_disparities,
        proc_p
    ) = procrustes_permutation_test(
        pos_pred,
        pos_pram,
        n_permutations=(
            N_PERMUTATIONS
        ),
        random_state=(
            PERMUTATION_RANDOM_STATE
        )
    )

    # ======================================
    # Print
    # ======================================

    print()
    print(
        "=" * 70
    )

    print(
        f"{label}"
    )

    print(
        "=" * 70
    )

    print(
        f"Pearson r                  : "
        f"{r:.4f}"
    )

    print(
        f"Pearson p                  : "
        f"{r_p:.6f}"
    )

    print(
        f"Spearman rho               : "
        f"{rho:.4f}"
    )

    print(
        f"Spearman p                 : "
        f"{rho_p:.6f}"
    )

    print(
        f"Prediction Stress          : "
        f"{stress_pred:.5f}"
    )

    print(
        f"PRaM Stress                : "
        f"{stress_pram:.5f}"
    )

    print(
        f"Procrustes disparity       : "
        f"{disparity:.5f}"
    )

    print(
        f"Permutation p              : "
        f"{proc_p:.6f}"
    )

    print(
        f"Null disparity mean        : "
        f"{null_disparities.mean():.5f}"
    )

    print(
        f"Null disparity SD          : "
        f"{null_disparities.std():.5f}"
    )

    print(
        f"Null disparity min         : "
        f"{null_disparities.min():.5f}"
    )

    print(
        f"Null disparity max         : "
        f"{null_disparities.max():.5f}"
    )

    return {
        "aligned_pred":
            aligned_pred,

        "aligned_pram":
            aligned_pram,

        "raw_pred":
            pos_pred,

        "raw_pram":
            pos_pram,

        "pearson_r":
            r,

        "pearson_p":
            r_p,

        "spearman_rho":
            rho,

        "spearman_p":
            rho_p,

        "stress_pred":
            stress_pred,

        "stress_pram":
            stress_pram,

        "disparity":
            disparity,

        "permutation_p":
            proc_p,

        "null_disparities":
            null_disparities,
    }


# ==========================================
# MDS results save
# ==========================================

def save_mds_results(
    pos_pred,
    pos_pram,
    output_dir="average_results",
):

    os.makedirs(
        output_dir,
        exist_ok=True
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
            "Item":
                labels,

            "X":
                pos_pred[:, 0],

            "Y":
                pos_pred[:, 1]
        }
    )

    coord_pram = pd.DataFrame(
        {
            "Item":
                labels,

            "X":
                pos_pram[:, 0],

            "Y":
                pos_pram[:, 1]
        }
    )

    coord_pred.to_csv(
        os.path.join(
            output_dir,
            "mds_prediction_coordinates.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    coord_pram.to_csv(
        os.path.join(
            output_dir,
            "mds_pram_coordinates.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------
    # Distance matrices
    # --------------------------------------

    dist_pred = squareform(
        pdist(
            pos_pred
        )
    )

    dist_pram = squareform(
        pdist(
            pos_pram
        )
    )

    pd.DataFrame(
        dist_pred,
        index=labels,
        columns=labels
    ).to_csv(
        os.path.join(
            output_dir,
            "mds_prediction_distance.csv"
        ),
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        dist_pram,
        index=labels,
        columns=labels
    ).to_csv(
        os.path.join(
            output_dir,
            "mds_pram_distance.csv"
        ),
        encoding="utf-8-sig"
    )

    print(
        "\nMDS coordinates and "
        "distance matrices saved."
    )


# ==========================================
# Permutation results save
# ==========================================

def save_permutation_results(
    result,
    output_dir="average_results",
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    null_disparities = result[
        "null_disparities"
    ]

    # --------------------------------------
    # Distribution
    # --------------------------------------

    df_null = pd.DataFrame(
        {
            "Permutation":
                np.arange(
                    1,
                    len(
                        null_disparities
                    )
                    +
                    1
                ),

            "Disparity":
                null_disparities
        }
    )

    df_null.to_csv(
        os.path.join(
            output_dir,
            "prediction_procrustes_permutation_distribution.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------
    # Summary
    # --------------------------------------

    summary = pd.DataFrame(
        [
            {
                "Observed_disparity":
                    result[
                        "disparity"
                    ],

                "Permutation_p":
                    result[
                        "permutation_p"
                    ],

                "Null_mean":
                    null_disparities.mean(),

                "Null_SD":
                    null_disparities.std(),

                "Null_min":
                    null_disparities.min(),

                "Null_max":
                    null_disparities.max(),

                "N_permutations":
                    len(
                        null_disparities
                    )
            }
        ]
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            "prediction_procrustes_permutation_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


# ==========================================
# 実行
# ==========================================

print(
    "\n--- Running Analysis ---"
)

pram_f = np.array(
    pram_matrices
)[
    mask
]

result = compute_metrics(
    prediction_matrix,
    pram_f,
    "Prediction vs PRaM"
)


# ==========================================
# Average matrices save
# ==========================================

save_average_matrices(
    prediction_matrix,
    pram_f,
    output_dir="average_results"
)


# ==========================================
# MDS save
# ==========================================

save_mds_results(
    result[
        "raw_pred"
    ],
    result[
        "raw_pram"
    ],
    output_dir="average_results"
)


# ==========================================
# Permutation save
# ==========================================

save_permutation_results(
    result,
    output_dir="average_results"
)


# ==========================================
# Permutation plot
# ==========================================

plot_permutation_distribution(
    result[
        "null_disparities"
    ],
    result[
        "disparity"
    ]
)


# ==========================================
# MDS visualization
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8)
)

plot_with_images(
    axes[0],
    result[
        "aligned_pred"
    ],
    "Predicted PRaM"
)

plot_with_images(
    axes[1],
    result[
        "aligned_pram"
    ],
    (
        f"Measured PRaM "
        f"(N={len(pram_f)})"
    )
)

plt.suptitle(
    "Prediction vs Measured PRaM"
)

plt.tight_layout()

plt.show()


# ==========================================
# Final summary
# ==========================================

print()
print(
    "=" * 70
)

print(
    "FINAL RESULT"
)

print(
    "=" * 70
)

print(
    f"Pearson r            : "
    f"{result['pearson_r']:.3f}"
)

print(
    f"Spearman rho         : "
    f"{result['spearman_rho']:.3f}"
)

print(
    f"Prediction Stress    : "
    f"{result['stress_pred']:.3f}"
)

print(
    f"PRaM Stress          : "
    f"{result['stress_pram']:.3f}"
)

print(
    f"Procrustes disparity : "
    f"{result['disparity']:.3f}"
)

print(
    f"Permutation p        : "
    f"{result['permutation_p']:.6f}"
)

print()

print(
    "Analysis Complete."
)
