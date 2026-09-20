import glob
import os
import re

import numpy as np
import pandas as pd

# =========================================================
# PRaM Reader
# =========================================================

def read_pram_folder(folder):
    """
    Read all PRaM csv files.

    Returns
    -------
    DataFrame

    Subject
    A
    B
    Resp
    """

    files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    all_data = []

    for subject_id, file in enumerate(files):

        df = pd.read_csv(file)

        rows = []

        for _, row in df.iterrows():

            image_name = os.path.basename(str(row["images_ocha_A"]))

            match = re.search(r"(\d+)_(\d+)", image_name)

            if match is None:
                continue

            a = int(match.group(1))
            b = int(match.group(2))

            if a > b:
                a, b = b, a

            rows.append(
                {
                    "Subject": subject_id,
                    "A": a,
                    "B": b,
                    "Resp": int(row["key_resp_2.keys"])
                }
            )

        all_data.append(pd.DataFrame(rows))

    pram = pd.concat(all_data, ignore_index=True)

    return pram


# =========================================================
# SpAM Reader
# =========================================================

def read_one_spam(file):
    """
    Read one SpAM csv.

    Distance Matrix以降だけ読み込む。
    """

    with open(file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    start = None

    for i, line in enumerate(lines):
        if "Distance Matrix:" in line:
            start = i + 1
            break

    if start is None:
        raise ValueError(f"Distance Matrix not found: {file}")

    matrix = []

    for line in lines[start:]:

        line = line.strip()

        if line == "":
            continue

        row = [float(x) for x in line.split(",") if x != ""]

        matrix.append(row)

    return np.array(matrix)


def read_spam_folder(folder):
    """
    Read all SpAM csv files.

    Returns
    -------
    list(ndarray)
    """

    files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    matrices = []

    for file in files:

        matrices.append(read_one_spam(file))

    return matrices


# =========================================================
# Utilities
# =========================================================

def average_spam(matrices):
    """
    Average distance matrices.
    """

    return np.mean(matrices, axis=0)