# Subject-based SpAM-to-PRaM prediction

This code evaluates whether a relationship learned between SpAM distance and
PRaM rating generalizes to held-out participants. Participants, rather than
stimuli, are divided into training and test groups.

## Files to run

| File | Purpose | Main output |
| --- | --- | --- |
| `main_compare_models.py` | Compare seven models over participant training sizes | Performance CSV files |
| `main_prediction.py` | Fit Ordered Logit using a fixed participant split | Pair-level PRaM predictions and participant split |
| `main_mds_analysis.py` | Run both MDS/Procrustes comparisons | MDS coordinates, distance matrices, and permutation results |

The other Python files contain the data-loading, model, evaluation, experiment,
and optional visualization functions used by the workflow.

## Data layout

Run the code from a directory containing:

```text
subject_based_prediction_github/
├─ PRaM_data/
│  └─ participant CSV files
├─ SpAM_data/
│  └─ participant CSV files
├─ image/
│  └─ 1.png ... 16.png        # optional, for MDS figures
└─ Python files
```

PRaM and SpAM files must sort in the same participant order because each
reader assigns integer `Subject` IDs from the sorted file order.

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## 1. Model comparison

```bash
python main_compare_models.py
```

Defaults:

- 500 Monte Carlo trials;
- 4–15 training participants;
- Ordered Logit, Ordered Probit, Linear, Lasso, Random Forest, XGBoost, and KNN.

Outputs:

```text
results/subject_performance_curve.csv
results/subject_performance_summary.csv
```

## 2. Fixed-split prediction

```bash
python main_prediction.py
```

Defaults:

- 10 training participants;
- remaining participants used for testing;
- random seed 42;
- Ordered Logit model.

Outputs:

```text
results/PRaM_prediction.csv
results/subject_split.csv
```

## 3. MDS analyses

Run `main_prediction.py` first, then:

```bash
python main_mds_analysis.py
```

The single script runs two analyses in sequence:

1. predicted PRaM versus measured mean PRaM;
2. mean SpAM versus predicted PRaM.

Outputs are written to:

```text
average_results/
spam_vs_prediction_results/
```

## Accuracy within ±1

Accuracy within ±1 is calculated without rounding:

```python
abs(measured_mean_pram - predicted_pram) <= 1.0
```
