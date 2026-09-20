# SpAM2PRaM-From-Distance-to-Scale
SpAM2PRaM is a Python-based framework for predicting Pairwise Rating Method (PRaM) scale values from distance data obtained using the Spatial Arrangement Method (SpAM).

SpAM represents perceived similarity as spatial distances between stimuli, whereas PRaM records similarity directly on a numerical rating scale. This repository links these two forms of similarity measurement by learning a function that predicts 1–7 PRaM similarity ratings from SpAM distance data.

The broader objective is to reduce the burden of exhaustive pairwise rating while retaining an interpretable scale-based measure of perceived similarity.

# Overview
SpAM allows participants to represent perceived similarity by freely arranging multiple stimuli in a two-dimensional space. The Euclidean distances between the arranged stimuli provide an efficient representation of their overall similarity structure.

However, SpAM produces spatial distances rather than conventional rating-scale values. By contrast, PRaM directly measures the perceived similarity of each stimulus pair using a 1–7 scale.

SpAM2PRaM uses SpAM distance as the explanatory variable and PRaM rating as the target variable:
’’’
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
’’’
