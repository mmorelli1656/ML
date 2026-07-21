# -*- coding: utf-8 -*-
"""
example_usage_simple.py
Flat script version: no functions, everything at module level.
"""

import os
import sys

ML_ROOT = r"C:\Users\WKS\Github\ML"
sys.path.insert(0, ML_ROOT)
os.environ["PYTHONPATH"] = ML_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

from trainer_v3 import ParallelModelTrainer


# ------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------
N_SPLITS = 10
N_REPEATS = 20
RANDOM_STATE = 42

SCALER = StandardScaler()

MODEL = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    eval_metric="mlogloss",
)

BALANCER = None
FEATURE_SELECTORS = [SelectKBest(score_func=f_classif, k=10)]
CLASSES_TO_SAVE = [1, 2]
N_CORES = -1
MASTER_SEED = 42

DEBUG_SEQUENTIAL = False   # <-- QUESTA è la riga importante per il test di oggi


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
X, y = make_classification(
    n_samples=300, n_features=20, n_informative=8,
    n_classes=3, n_clusters_per_class=1, random_state=0,
)
X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="target")


# ------------------------------------------------------------------
# TRAINER SETUP + RUN
# ------------------------------------------------------------------
rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

trainer = ParallelModelTrainer(
    X=X, y=y, rkf=rkf, scaler=SCALER, model=MODEL, balancer=BALANCER,
    feature_selectors=FEATURE_SELECTORS, classes_to_save=CLASSES_TO_SAVE,
    n_cores=N_CORES, master_seed=MASTER_SEED,
)

if DEBUG_SEQUENTIAL:
    print("\n[DEBUG] Running folds sequentially (no joblib/loky)...")
    n_splits = rkf.get_n_splits(trainer.X, trainer.y)
    seeds = trainer._generate_seeds(n_splits)
    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(trainer.X, trainer.y)):
        print(f"--- Fold {fold_idx} ---")
        r = trainer.process_fold(fold_idx, train_idx, val_idx, seeds[fold_idx])
        results.append(r)
        print(f"Fold {fold_idx} completato.")
else:
    results = trainer.parallel_training()

outputs = trainer.get_all(results)

predictions = outputs["predictions"]
predictions_proba = outputs["predictions_proba"]
feature_selection = outputs["feature_selection"]
feature_importances = outputs["feature_importances"]
eval_history = outputs["eval_history"]
scaler_model = outputs["scaler_model"]

print("\n[OUTPUTS]")
print("Predictions shape:         ", predictions.shape)
print("Feature selection shape:   ", feature_selection.shape)
print("Feature importances shape: ", feature_importances.shape)
print("Eval history folds:        ", len(eval_history))



