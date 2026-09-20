# SpAM2PRaM-From-Distance-to-Scale
SpAM2PRaM is a Python-based framework for predicting Pairwise Rating Method (PRaM) scale values from distance data obtained using the Spatial Arrangement Method (SpAM).

SpAM represents perceived similarity as spatial distances between stimuli, whereas PRaM records similarity directly on a numerical rating scale. This repository links these two forms of similarity measurement by learning a function that predicts 1–7 PRaM similarity ratings from SpAM distance data.

The broader objective is to reduce the burden of exhaustive pairwise rating while retaining an interpretable scale-based measure of perceived similarity.

# Overview
SpAM allows participants to represent perceived similarity by freely arranging multiple stimuli in a two-dimensional space. The Euclidean distances between the arranged stimuli provide an efficient representation of their overall similarity structure.

However, SpAM produces spatial distances rather than conventional rating-scale values. By contrast, PRaM directly measures the perceived similarity of each stimulus pair using a 1–7 scale.

SpAM2PRaM uses SpAM distance as the explanatory variable and PRaM rating as the target variable:
```
SpAM distance
      ↓
Prediction model
      ↓
Predicted PRaM rating

Seven prediction models are compared:

Ordered Logistic Regression
Ordered Probit Regression
Linear Regression
Lasso Regression
Random Forest
XGBoost
K-Nearest Neighbors
```

Model performance is evaluated using continuous, non-rounded predictions with the following metrics:
```
・Pearson correlation
・Spearman’s rank correlation
・Kendall’s rank correlation
・Root Mean Squared Error (RMSE)
・Mean Absolute Error (MAE)
・Coefficient of determination (R²)
・Accuracy within ±1 scale point
```

Accuracy within ±1 is defined as:
```
abs(observed_pram - predicted_pram) <= 1.0
```

## Methodological background
The methodological basis of this repository follows the Spatial Arrangement Method introduced by Hout et al. (2013).

Hout and colleagues developed SpAM as an efficient alternative to exhaustive pairwise similarity judgments and demonstrated that similarity structures derived from spatial arrangements correspond closely to those obtained through traditional pairwise ratings.

In this repository:
```
・SpAM refers to similarity information represented by inter-stimulus spatial distances.
・PRaM refers to similarity information collected through direct pairwise ratings on a 1–7 scale.
・SpAM2PRaM predicts PRaM scale values from SpAM distance data.
```
> The relationship between spatial arrangement data and conventional pairwise similarity judgments is described in:
> Hout, M. C., Goldinger, S. D., & Ferguson, R. W. (2013). SpAM: A method for efficient 
> multidimensional scaling of similarity judgments. Behavior Research Methods, 45, 619–636.
> https://doi.org/10.3758/s13428-012-0292-6

Before using this repository, readers are encouraged to consult this article for the theoretical basis of SpAM, its relationship with pairwise similarity ratings, and its application to multidimensional similarity structures.

# Validation Strategies
This repository contains two complementary validation approaches.

## 1. Stimulus-based prediction
The stimulus-based analysis evaluates whether PRaM ratings for unmeasured stimulus pairs can be predicted from a limited subset of measured stimuli.

A subset of stimuli is randomly selected for model training. PRaM ratings for pairs formed by the selected stimuli are used as the training data, while ratings for the remaining stimulus pairs are predicted from their SpAM distances.

## 2. Participant-based prediction
The participant-based analysis evaluates whether the relationship learned between SpAM distance and PRaM rating generalizes to previously unseen participants.

Participants are randomly divided into training and test groups. The models are fitted using data from the training participants and evaluated against the mean PRaM ratings of the held-out test participants.

# Repository Structure
```
SpAM2PRaM/
├── stimulus-based/
│ ├── main_compare_models.py
│ ├── main_prediction.py
│ ├── main_mds_analysis.py
│ └── ...
│
└── participant-based/
├── main_compare_models.py
├── main_prediction.py
├── main_mds_analysis.py
└── ...
```
Each directory is self-contained and includes the modules required for data loading, model fitting, evaluation, visualization, and result export.

# Usage
Two analysis workflows are provided.

## Stimulus-based analysis
| Purpose | File |
| --- | --- |
| Compare prediction models across different numbers of training stimuli | `main_compare_models.py` |
| Predict PRaM ratings for unmeasured stimulus pairs | `main_prediction.py` |
| Compare predicted and measured similarity structures using MDS and Procrustes analysis | `mds_analysis.py` |

Recommended order:
```
python main_compare_models.py
python main_prediction.py
python mds_analysis.py
```

## Participant-based analysis
| Purpose | File |
| --- | --- |
| Compare prediction models across different numbers of training participants | `main_compare_models.py` |
| Predict PRaM ratings for held-out participants | `main_prediction.py` |
| Run the MDS and Procrustes analyses | `mds_analysis.py` |

Recommended order:
```
python main_compare_models.py
python main_prediction.py
python main_mds_analysis.py
```

# Data Preparation

## PRaM data
Place one CSV file per participant in:

| `PRaM_data/` |

Each CSV file must contain the following columns:

| `images_ocha_A`|
| `key_resp_2.keys`|

Example:

```
images_ocha_A,key_resp_2.keys
1_2.png,6
1_3.png,4
1_4.png,5
```

The stimulus pair is identified from the A_B numbers in the filename, and key_resp_2.keys contains the 1–7 PRaM similarity rating.

## SpAM data
Place one CSV file per participant in:

| `SpAM_data/`|

Each CSV file must contain a square distance matrix after the following marker:

Distance Matrix:

Example:
```
Distance Matrix:
0,120.4,350.2
120.4,0,180.5
350.2,180.5,0
```

For 16 stimuli, each file must contain a 16 × 16 distance matrix.

For the participant-based analysis, the PRaM and SpAM filenames must sort in the same participant order. Using matching filenames is recommended:
```
PRaM_data/P01.csv
SpAM_data/P01.csv

PRaM_data/P02.csv
SpAM_data/P02.csv
```

## Using a Different Data Format

If the input CSV format differs from the format described above, modify the data-loading functions while preserving the internal column names used by the analysis.

| Data to modify | File | Function |
| --- | --- | --- |
| PRaM filename, pair-ID pattern, rating column, or CSV structure | `data.py` | `read_pram_folder()` |
| SpAM participant-file processing | `data.py` | `read_one_spam()` |
| PRaM or SpAM directory path | `data.py` | `read_spam_folder()` |
| PRaM or SpAM directory path | `config.py` | `PRAM_DIR` and `SPAM_DIR` |

The modified readers should continue to return the following internal formats.
```
PRaM:

Subject, A, B, Resp

SpAM:

Subject, A, B, Distance
```
If only the original CSV column names differ, it is generally sufficient to edit data.py. The remaining model and evaluation files do not need to be modified as long as these internal column names are preserved.

The MDS scripts contain separate data-loading functions. If the raw-data or prediction-file format is changed, the corresponding MDS loader must also be updated.

| Workflow | File | Functions to modify |
| --- | --- | --- |
| Stimulus-based MDS | `mds_analysis.py` | `load_pram_matrix()` and `load_prediction_matrix()` |
| Participant-based MDS | `main_mds_analysis.py` | `load_pram_matrix()`, `load_spam_matrix()`, and `load_prediction_matrix()` |

For example, if the PRaM rating column is named rating instead of `key_resp_2.keys`, replace:

row`["key_resp_2.keys"]`

with:

row`["rating"]`

in the relevant loading function.


# Prerequisites

## Software requirements
The analyses require Python 3.10 or later.

The main Python dependencies are:
```
・NumPy
・pandas
・SciPy
・scikit-learn
・statsmodels
・XGBoost
・Matplotlib
・Pillow
```
Install the required packages using:
```
python -m pip install -r requirements.txt
```

# Data
The current analysis was developed using similarity judgments for 16 Japanese green-tea package designs.

```
・SpAM data consist of participant-level inter-stimulus distance matrices.
・PRaM data consist of participant-level pairwise similarity ratings on a 1–7 scale.
```

Raw participant data are not included in this repository. The expected data structure and execution instructions are described in the README file within each analysis directory.

# MDS and Structural Comparison
Predicted and observed similarity structures can also be compared using non-metric multidimensional scaling.

The structural analyses include:
```
・Non-metric multidimensional scaling
・MDS Stress
・Pearson and Spearman correlations between inter-stimulus distances
・Procrustes analysis
・Procrustes permutation tests
```

These analyses evaluate whether the predicted ratings reproduce not only individual rating values but also the broader perceptual structure among stimuli.

# Citation
If you use this repository in academic work, please cite: It has been accepted and will be posted as soon as it is made public.

# BibTeX
To be posted after a decision is made.

# Purpose
SpAM2PRaM bridges spatial similarity representation and conventional rating-scale measurement.

By deriving interpretable scale values from spatial distance data, the framework may reduce the burden associated with exhaustive pairwise evaluation while preserving the underlying perceptual relationships among stimuli.

From Distance to Scale.
