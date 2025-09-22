# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:14:05 2025

@author: mik16
"""

#%% Libreries and modules

import sys
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

# Import modules
my_ML_path = r"C:\Users\mik16\Github\ML"
sys.path.append(my_ML_path)

my_Utils_path = r"C:\Users\mik16\Github\Utils"
sys.path.append(my_Utils_path)

from parallel_gridsearch_v3 import ParallelGridSearch
from my_featsel import FeaturesVariance, FeaturesPearson
from elapsed_timer import Timer

del my_ML_path, my_Utils_path


#%% Load data

# Example dataset for regression
X, y = make_regression(
    n_samples=1000,      # number of samples
    n_features=40,       # number of features
    n_informative=10,    # number of informative features
    noise=0.1,           # noise added to the target variable
    random_state=42
)

X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y)


#%% Gridsearch parameters

# Repeated CV
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Pipeline with scaling, SMOTE, feature selection, model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ('var_fs', FeaturesVariance(mode='percentile')),
    ('pearson_fs', FeaturesPearson(threshold=0.80, alpha=0.01, random_state=42)),
    ("model", XGBRegressor(eval_metric="rmse", random_state=42))
])

# Parameters grid
param_grid = {
    "var_fs__threshold_value": [30, 70],
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5],
    'model__learning_rate': [0.01, 0.1, 0.2],
}

combinations = list(ParameterGrid(param_grid))
print(f"Number of parameters combinations: {len(combinations)}")

# Setup ParallelGridSearch
gs = ParallelGridSearch(
    X, y, rkf,
    estimator=pipe,
    param_grid=param_grid,
    classes_to_save=None,
    parallel_over="param",
    save=["model", "scaler", "features"]
)


#%% Gridsearch

# Parallel gridsearch
with Timer():
    results = gs.parallel_training()


#%% Aggregate results

# Performance metrics
metrics = {
    "r2": r2_score,          # coefficient of determination
    "mse": mean_squared_error, # mean squared error
    "mae": mean_absolute_error # mean absolute error
}

# All results
with Timer():
    df_results = gs.aggregate_results(results, metrics, use_proba=True)


#%% Best results

# Assume 'metrics' dictionary is already defined
while True:
    metric = input(f"Select a metric from {list(metrics.keys())}: ").strip()
    if metric in metrics:
        break
    print(f"Invalid choice. Please choose one of {list(metrics.keys())}.")

# Best results for selected metric
best_params = gs.get_best_params(df_results, metric=metric)
best_score = gs.get_best_score(df_results, metric=metric)

# Best parameters and scores 
print(f"Best parameters for {metric}:", best_params)
print(f"Best {metric}:", best_score)
