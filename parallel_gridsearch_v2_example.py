# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 18:42:25 2025

@author: mik16
"""

#%% Libreries and modules

import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid

# Import modules
my_ML_path = r"C:\Users\mik16\Github\ML"
sys.path.append(my_ML_path)

my_Utils_path = r"C:\Users\mik16\Github\Utils"
sys.path.append(my_Utils_path)

from parallel_gridsearch_v2 import ParallelGridSearch
from elapsed_timer import Timer

del my_ML_path, my_Utils_path


#%% Load data

# Example dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


#%% Gridsearch parameters

# Repeated CV
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)

# Scaler
scaler = StandardScaler()

# Model
model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# Parameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

combinations = list(ParameterGrid(param_grid))
print(f"Numero di combinazioni di parametri: {len(combinations)}")

# Setup ParallelGridSearch
grid_search = ParallelGridSearch(
    X=X,
    y=y,
    rskf=rskf,
    scaler=scaler,
    model=model,
    param_grid=param_grid,
    balancer=None,
    feature_selectors=[],  # puoi testare con selettori anche
    classes_to_save=[1],  # per AUC, classe positiva
    n_cores=-1,  # usa tutti i core
    parallel_over="param",
    save_models=False
)


#%% Gridsearch

# Parallel gridsearch
with Timer():
    results = grid_search.parallel_training()


#%% Aggregate results

# Performance metrics
metrics = {
    "auc": roc_auc_score,
    "recall": recall_score,
    "precision": precision_score
}

# All results
with Timer():
    df_results = grid_search.aggregate_results(results, metrics, use_proba=True)
    

#%% Best reuslts

# Best results for selected metric
best_params = grid_search.get_best_params(df_results, metric="auc")
best_score = grid_search.get_best_score(df_results, metric="auc")

# Show results
print(df_results)