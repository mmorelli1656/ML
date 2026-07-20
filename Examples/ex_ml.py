# -*- coding: utf-8 -*-
"""
example_usage.py
Minimal runnable example showing how to use ParallelModelTrainer + adapters.

Covers:
- A standard XGBoost multiclass setup (feature selection, scaling,
  classes_to_save, repeated stratified CV).
- How the new validation guards behave (classes_to_save, n_cores=0).
- A fix for BrokenProcessPool / "task has failed to un-serialize" errors
  that can occur on Windows when running under an embedded interpreter
  (e.g. Spyder's IPython console), where sys.executable may not point
  to the actual environment's python.exe that loky needs to spawn workers.

Requires: scikit-learn, xgboost, joblib, tqdm, pandas, numpy
(the same stack you already use).
"""

import sys
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

from trainer_v3 import ParallelModelTrainer


def build_dataset():
    """Synthetic 3-class dataset, just to make the example self-contained."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=8,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


def main():
    X, y = build_dataset()

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    trainer = ParallelModelTrainer(
        X=X,
        y=y,
        rkf=rkf,
        scaler=StandardScaler(),
        model=XGBClassifier(
            n_estimators=100,
            max_depth=3,
            eval_metric="mlogloss",
        ),
        balancer=None,  # e.g. imblearn.over_sampling.SMOTE() if needed
        feature_selectors=[SelectKBest(score_func=f_classif, k=10)],
        classes_to_save=[1, 2],  # valid: dataset has classes 0, 1, 2
        n_cores=-1,              # use all available cores
        master_seed=42,
    )

    print("\n[RUN] Starting parallel repeated cross-validation...")
    results = trainer.parallel_training()

    outputs = trainer.get_all(results)

    print("\n[OUTPUTS]")
    print("Predictions shape:         ", outputs["predictions"].shape)
    proba = outputs["predictions_proba"]
    if isinstance(proba, list):
        print("Predictions (proba) shapes:", [df.shape for df in proba])
    else:
        print("Predictions (proba) shape: ", proba.shape)
    print("Feature selection shape:   ", outputs["feature_selection"].shape)
    print("Feature importances shape: ", outputs["feature_importances"].shape)
    print("Eval history folds:        ", len(outputs["eval_history"]))
    print("\nHead of predictions:")
    print(outputs["predictions"].head())


def demo_validation_guards():
    """Shows the new fail-fast behavior for invalid inputs (no training run)."""
    X, y = build_dataset()
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

    print("\n[DEMO] classes_to_save out of range:")
    try:
        ParallelModelTrainer(
            X=X, y=y, rkf=rkf, scaler=StandardScaler(),
            model=XGBClassifier(), classes_to_save=[5],  # only classes 0,1,2 exist
        )
    except ValueError as e:
        print(f"  -> Raised as expected: {e}")

    print("\n[DEMO] n_cores=0:")
    try:
        ParallelModelTrainer(
            X=X, y=y, rkf=rkf, scaler=StandardScaler(),
            model=XGBClassifier(), n_cores=0,
        )
    except ValueError as e:
        print(f"  -> Raised as expected: {e}")


if __name__ == "__main__":
    # Fix for BrokenProcessPool on Windows / embedded interpreters (e.g. Spyder):
    # ensures loky spawns workers using the correct python.exe for this
    # environment, instead of whatever sys.executable might otherwise resolve to.
    multiprocessing.set_executable(sys.executable)

    demo_validation_guards()
    main()