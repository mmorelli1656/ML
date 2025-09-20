# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 17:43:54 2025

@author: mik16
"""

#%% Libreries and modules

import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

# Import modules
my_Utils_path = r"C:\Users\mik16\Github\Utils"
sys.path.append(my_Utils_path)

from elapsed_timer import Timer

del my_Utils_path


#%% Load data

# Example dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


#%% Gridsearch parameters

# Repeated CV
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)

# Model
model = XGBClassifier(
    eval_metric='logloss',  # metrica interna
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])

# Parameters grid
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.7, 1.0],
    'model__colsample_bytree': [0.7, 1.0]
}

combinations = list(ParameterGrid(param_grid))
print(f"Numero di combinazioni di parametri: {len(combinations)}")

# Performance metrics
scoring = {
    'auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision'
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit='auc',  # the most important metric
    cv=rskf,
    n_jobs=-1,
    verbose=2
)

# Gridsearch
with Timer():
    grid_search.fit(X, y)

# Best parameters and scores (refit)
print("Migliori parametri secondo AUC:", grid_search.best_params_)
print("Miglior AUC:", grid_search.best_score_)

# All results
results = grid_search.cv_results_
df_results = pd.DataFrame(results)

print(df_results[['params', 'mean_test_auc', 'mean_test_recall', 'mean_test_precision']])