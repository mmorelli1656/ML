# -*- coding: utf-8 -*-
"""
adapters.py
Model adapters for ParallelModelTrainer.

Provides a generic interface (ModelAdapter) for training and metadata
extraction, plus a specialized adapter for XGBoost models that supports
eval_set / eval_history tracking.

XGBoost is treated as an optional dependency: if it is not installed,
_HAS_XGB is False and XGBAdapter simply becomes unavailable, without
breaking the rest of the module.

Author: mik16
"""

import pandas as pd

try:
    from xgboost import XGBClassifier, XGBRegressor  # noqa: F401
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ------------------------------------------------------------------
# Generic adapter
# ------------------------------------------------------------------

class ModelAdapter:
    """Generic interface for model-specific training and metadata extraction."""

    def fit(self, model, X_train, y_train, X_val=None, y_val=None, seed=None):
        """Train the model and return (trained_model, eval_history)."""
        if seed is not None and hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        model.fit(X_train, y_train)
        return model, None

    def get_feature_importances(self, model, feature_names):
        """Return feature importances if available."""
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=feature_names)
        return None


# ------------------------------------------------------------------
# XGBoost adapter
# ------------------------------------------------------------------

class XGBAdapter(ModelAdapter):
    """Adapter for XGBoost models (supports eval history)."""

    def fit(self, model, X_train, y_train, X_val=None, y_val=None, seed=None):
        if seed is not None:
            params = dict(
                random_state=seed,
                seed=seed,
                n_jobs=1,
            )
            if "deterministic_histogram" in model.get_params():
                params["deterministic_histogram"] = True
            model.set_params(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)] if X_val is not None else None,
            verbose=False,
        )
        evals_result = model.evals_result() if hasattr(model, "evals_result") else None
        return model, evals_result