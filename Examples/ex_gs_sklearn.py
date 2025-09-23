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
from pathlib import Path

# Import modules
sys.path.append(str(Path.home() / "Github" / "ML"))
sys.path.append(str(Path.home() / "Github" / "Utils"))

from elapsed_timer import Timer
from my_featsel import FeaturesVariance, FeaturesPearson


#%% Load data

# Example dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


#%% Gridsearch parameters

# Repeated CV
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Model
model = XGBClassifier(
    eval_metric='logloss',  # metrica interna
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('var_fs', FeaturesVariance(threshold_value=50, mode='percentile')),
    ('pearson_fs', FeaturesPearson(threshold=0.80, alpha=0.01, random_state=42)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(eval_metric='logloss', random_state=42))
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
    verbose=1
)


#%% Gridsearch

# Gridsearch
with Timer():
    grid_search.fit(X, y)


#%% Aggregate results

# All results
results = grid_search.cv_results_
df_results = pd.DataFrame({
    'params': results['params'],
    'mean_test_auc': results['mean_test_auc'],
    'std_test_auc': results['std_test_auc'],
    'mean_test_recall': results['mean_test_recall'],
    'std_test_recall': results['std_test_recall'],
    'mean_test_precision': results['mean_test_precision'],
    'std_test_precision': results['std_test_precision'],
})


#%% Best results

# Best results for selected metric (refit)
print("Best parameters AUC:", grid_search.best_params_)
print("Best AUC:", grid_search.best_score_)