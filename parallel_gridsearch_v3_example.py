# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 09:06:06 2025

@author: mik16
"""

#%% Libreries and modules

import sys
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, roc_auc_score, precision_score
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

# Example dataset for classification (unbalanced)
X, y = make_classification(
    n_samples=1000,      # total number of samples (rows in the dataset)
    n_features=40,       # total number of features (columns)
    n_informative=10,    # number of informative features (useful to distinguish the classes)
    n_redundant=5,       # number of redundant features (linear combinations of informative ones)
    weights=[0.7, 0.3],  # class distribution (80% class 0, 20% class 1)
    random_state=42      # random seed for reproducibility
)

X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y)


#%% Gridsearch parameters

# Repeated CV
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

# Pipeline with scaling, SMOTE, feature selection, model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ('var_fs', FeaturesVariance(threshold_value=50, mode='percentile')),
    ('pearson_fs', FeaturesPearson(threshold=0.80, alpha=0.01, random_state=42)),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42))
])

# Parameters grid
param_grid = {
    "smote__sampling_strategy": [0.5, 1.0],
    "var_fs__threshold_value": [30, 70],
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5],
    'model__learning_rate': [0.01, 0.1, 0.2],
}

combinations = list(ParameterGrid(param_grid))
print(f"Number of parameters combinations: {len(combinations)}")

# Setup ParallelGridSearch
gs = ParallelGridSearch(
    X, y, rskf,
    estimator=pipe,
    param_grid=param_grid,
    classes_to_save=[1],
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
    "recall": recall_score,
    "auc": roc_auc_score,
    "precision": precision_score
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
