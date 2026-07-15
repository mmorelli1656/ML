# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:27:55 2026

@author: WKS
"""

# -*- coding: utf-8 -*-
"""
Test script for the refactored SHAPHandler / SHAPPlotter classes.

Covers:
- Building fold "results" objects (model, fold_idx, val_idx, selected_features, scaler)
  the way a CV pipeline would produce them.
- Sequential execution, parallel_level="fold", parallel_level="repeat".
- use_scaled=True/False, now passed to compute_shap_values() instead of the constructor.
- Reuse of the same configured SHAPHandler across two different result sets.
- SHAPPlotter reading state off the handler without needing X passed again.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend so plt.show() never blocks in a script/CI run

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from xai_shap import SHAPHandler, SHAPPlotter


class FoldResult:
    """Mimics whatever object your CV pipeline stores per fold."""
    def __init__(self, model, fold_idx, val_idx, selected_features, scaler=None):
        self.model = model
        self.fold_idx = fold_idx
        self.val_idx = val_idx
        self.selected_features = selected_features
        self.scaler = scaler


def make_dataset(n_samples=200, n_features=8, random_state=0):
    X_arr, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=1,
        random_state=random_state,
    )
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    return X, y


def build_results(X, y, rkf, model_cls, use_scaler=False, feature_subset=None):
    """
    Trains one model per (train, val) split, optionally fits a scaler on the
    training fold only, and packages everything into FoldResult objects —
    this is the "results" list that compute_shap_values() expects.
    """
    selected_features = feature_subset or list(X.columns)
    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(X, y)):
        X_train = X.iloc[train_idx][selected_features]
        y_train = y[train_idx]

        scaler = None
        if use_scaler:
            scaler = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(
                scaler.transform(X_train), columns=selected_features
            )

        model = model_cls().fit(X_train, y_train)
        results.append(
            FoldResult(
                model=model,
                fold_idx=fold_idx,
                val_idx=val_idx,
                selected_features=selected_features,
                scaler=scaler,
            )
        )
    return results


def check_shap_dict(shap_dict, X, n_repeats, label):
    assert set(shap_dict.keys()) == set(range(1, n_repeats + 1)), \
        f"[{label}] unexpected repetition keys: {shap_dict.keys()}"
    for r, df in shap_dict.items():
        assert isinstance(df, pd.DataFrame), f"[{label}] repeat {r} is not a DataFrame"
        assert df.index.isin(X.index).all(), f"[{label}] repeat {r} has out-of-range index"
        assert set(df.columns).issubset(set(X.columns)), \
            f"[{label}] repeat {r} has unexpected columns"
    print(f"[{label}] OK - {n_repeats} repetitions, "
          f"shapes: {[shap_dict[r].shape for r in shap_dict]}")


def main():
    X, y = make_dataset(n_samples=150, n_features=6, random_state=42)
    rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
    n_repeats = rkf.n_repeats

    results_tree = build_results(X, y, rkf, RandomForestClassifier, use_scaler=False)

    # --- Test 1: sequential execution, tree explainer ---
    handler = SHAPHandler(explainer_type="tree", parallel_level=None)
    shap_dict_seq = handler.compute_shap_values(results_tree, X, rkf, use_scaled=False)
    check_shap_dict(shap_dict_seq, X, n_repeats, "sequential/tree")

    # --- Test 2: parallel_level="fold" ---
    handler_fold = SHAPHandler(explainer_type="tree", n_jobs=2, parallel_level="fold")
    shap_dict_fold = handler_fold.compute_shap_values(results_tree, X, rkf, use_scaled=False)
    check_shap_dict(shap_dict_fold, X, n_repeats, "parallel_level=fold")

    # --- Test 3: parallel_level="repeat" ---
    handler_repeat = SHAPHandler(explainer_type="tree", n_jobs=2, parallel_level="repeat")
    shap_dict_repeat = handler_repeat.compute_shap_values(results_tree, X, rkf, use_scaled=False)
    check_shap_dict(shap_dict_repeat, X, n_repeats, "parallel_level=repeat")

    # Sanity check: sequential and parallel modes should agree on shape at least
    for r in range(1, n_repeats + 1):
        assert shap_dict_seq[r].shape == shap_dict_fold[r].shape == shap_dict_repeat[r].shape, \
            f"Shape mismatch across execution modes for repeat {r}"
    print("Sequential / fold-parallel / repeat-parallel all agree on output shape.\n")

    # --- Test 4: use_scaled=True with a linear explainer + scaler in results ---
    results_linear = build_results(
        X, y, rkf, LogisticRegression, use_scaler=True
    )
    handler_linear = SHAPHandler(explainer_type="linear", parallel_level=None)
    shap_dict_scaled = handler_linear.compute_shap_values(
        results_linear, X, rkf, use_scaled=True
    )
    check_shap_dict(shap_dict_scaled, X, n_repeats, "use_scaled=True/linear")

    # --- Test 5: reuse the same configured handler across two different result sets ---
    # (this is the whole point of moving results/X/rkf out of the constructor)
    results_tree_v2 = build_results(X, y, rkf, RandomForestClassifier, use_scaler=False)
    shap_dict_v2 = handler.compute_shap_values(results_tree_v2, X, rkf, use_scaled=False)
    check_shap_dict(shap_dict_v2, X, n_repeats, "handler reused on second result set")

    # --- Test 6: plotting via SHAPPlotter, reading state off the handler ---
    plotter = SHAPPlotter(handler, show=False)  # show=False: headless-friendly
    top_features = plotter.plot_summary_aggregated(max_display=5, save_path="/tmp")
    assert isinstance(top_features, pd.DataFrame)
    assert len(top_features) <= 5
    print("\n[plot_summary_aggregated] top features:")
    print(top_features)
    print("\nSaved plot to /tmp/shap_summary.png")

    # --- Test 7: plotter raises a clear error if called before compute_shap_values() ---
    fresh_handler = SHAPHandler()
    fresh_plotter = SHAPPlotter(fresh_handler)
    try:
        fresh_plotter.plot_summary_aggregated()
        raise AssertionError("Expected RuntimeError was not raised")
    except RuntimeError as e:
        print(f"\n[guard check] correctly raised: {e}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()