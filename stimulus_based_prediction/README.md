# SpAM–PRaM analysis code

This repository contains two related analyses:

1. prediction of PRaM ratings from SpAM distances;
2. comparison of SpAM, observed PRaM, and predicted PRaM structures using MDS.

The code was separated from `main.ipynb` into executable Python files. The
original module structure and calculations are otherwise retained.

## Files

- `main_compare_models.py`: compares all prediction models across training-set sizes.
- `main_prediction.py`: combines measured PRaM pairs and SpAM-based predictions.
- `main_trial.py`: runs and prints one trial.
- `plot_predictions.py`: plots observed versus predicted PRaM values.
- `mds_analysis.py`: independent MDS, Procrustes, and permutation-test analysis.
- `data.py`, `preprocessing.py`, `pipeline.py`: data preparation.
- `model.py`, `evaluation.py`, `experiment.py`, `visualization.py`: model fitting,
  evaluation, repeated experiments, and figures.

## Evaluation definition

`evaluation.py` is based on the original `evaluation_notround.py`.

- Spearman's rho, Kendall's tau, RMSE, MAE, and R-squared use continuous
  participant-mean PRaM scores and continuous predictions.
- Accuracy within ±1 is calculated without rounding:
  `abs(observed_mean - predicted) <= 1`.
- Exact rounded Accuracy is not calculated.

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Place the raw files under `PRaM_data/` and `SpAM_data/`, or edit their paths in
`config.py`.

## Model comparison

```bash
python main_compare_models.py
```

This runs the models for 4–8 training stimuli, writes trial-level and summary
CSV files, and displays the comparison figures.

## Completed PRaM data

Choose the prediction model in `main_prediction.py`, then run:

```bash
python main_prediction.py
```

The output is `results/PRaM_completed.csv`, containing measured training pairs
and predicted remaining pairs.

## MDS analysis

`mds_analysis.py` is intentionally separate from the machine-learning entry
points. Review the path and analysis settings at the top of that file before
running:

```bash
python mds_analysis.py
```

Raw participant data and generated results are excluded by `.gitignore`.
