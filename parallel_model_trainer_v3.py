# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 18:09:59 2025

@author: mik16
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 19:01:44 2025

@author: mik16
"""

# -*- coding: utf-8 -*-
"""
ParallelModelTrainer v2.0
Generic parallel cross-validation trainer with model adapters.

Supports any scikit-learn compatible estimator, and includes
an automatic adapter for XGBoost models to track feature importances
and evaluation history (train/validation loss).

Author: mik16
"""

import pandas as pd
import numpy as np
import random
from collections import namedtuple
from sklearn.base import clone, BaseEstimator
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from typing import List, Optional, Union, Tuple


# ------------------------------------------------------------------
# Model Adapters
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


# ------------------------------------------------------------------
# Parallel Model Trainer
# ------------------------------------------------------------------

class ParallelModelTrainer:
    """
    ParallelModelTrainer performs repeated cross-validation in parallel,
    supporting optional feature selection, balancing, scaling, and model training.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Union[pd.Series, np.ndarray]
        Target vector.
    rkf : object
        Cross-validation splitter (e.g., RepeatedKFold, RepeatedStratifiedKFold).
    scaler : BaseEstimator
        Scikit-learn compatible scaler (e.g., StandardScaler, MinMaxScaler).
    model : BaseEstimator
        Scikit-learn compatible model (e.g., LogisticRegression, XGBClassifier).
    balancer : Optional[BaseEstimator], default=None
        A resampling strategy (e.g., SMOTE).
    feature_selectors : Optional[List[BaseEstimator]], default=None
        A list of feature selection transformers.
    classes_to_save : Optional[List[int]], default=None
        Class indices for which to save predicted probabilities.
    n_cores : int, default=-1
        Number of parallel jobs for joblib. -1 uses all available cores.
    """

    FoldResult = namedtuple(
        "FoldResult",
        [
            "fold_idx",
            "val_idx",
            "selected_features",
            "scaler",
            "model",
            "y_pred",
            "y_pred_proba",
            "feature_importances",
            "eval_history",
        ],
    )

    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        rkf,
        scaler,
        model,
        balancer=None,
        feature_selectors: Optional[List[BaseEstimator]] = None,
        classes_to_save: Optional[List[int]] = None,
        n_cores: int = -1,
        master_seed: Optional[int] = 42,
    ):
        self.X = X
        self.y = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        self.rkf = rkf
        self.scaler = scaler
        self.model = model
        self.balancer = balancer
        self.feature_selectors = feature_selectors or []
        self.classes_to_save = classes_to_save
        self.n_cores = n_cores
        self.master_seed = master_seed
        self.rng = np.random.default_rng(master_seed)

        # Automatically select model adapter
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            self.model_adapter = XGBAdapter()
        else:
            self.model_adapter = ModelAdapter()
            
    # ------------------------------------------------------------------
    # Generate seeds
    # ------------------------------------------------------------------
    def _generate_seeds(self, n_splits: int) -> List[int]:
        """Generate a deterministic list of random seeds, one per fold."""
        return [int(s) for s in self.rng.integers(0, 2**32 - 1, size=n_splits)]

    # ------------------------------------------------------------------
    # Preprocessing per fold
    # ------------------------------------------------------------------
    def _apply_feature_selection(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Apply all feature selectors sequentially."""
        for selector in self.feature_selectors:
            sel = clone(selector)
            if hasattr(sel, "set_output"):
                sel.set_output(transform="pandas")
            X_train = sel.fit_transform(X_train, y_train)
            X_val = sel.transform(X_val)
        return X_train, X_val, X_train.columns.tolist()
    
    def _apply_balancer(
        self, X_train: pd.DataFrame, y_train: pd.Series, selected_features: List[str], seed: Optional[int]=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply class balancing (e.g., SMOTE) if provided. Clone and set seed on the clone."""
        if self.balancer is not None:
            balancer = clone(self.balancer)
            if seed is not None and hasattr(balancer, "random_state"):
                balancer.set_params(random_state=seed)
            X_res, y_res = balancer.fit_resample(X_train, y_train)
            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=selected_features)
            return X_res, y_res
        return X_train, y_train


    def _apply_scaler(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, BaseEstimator]:
        """Scale train and validation sets using the provided scaler."""
        scaler = clone(self.scaler)
        X_train_scaled = scaler.fit_transform(X_train, y_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    # ------------------------------------------------------------------
    # Training per fold
    # ------------------------------------------------------------------
    def _train_model(
        self, X_train_scaled: np.ndarray, y_train: np.ndarray,
        X_val_scaled: np.ndarray, y_val: np.ndarray, seed: int
    ) -> Tuple[BaseEstimator, Optional[dict]]:
        """Train the model using the appropriate adapter."""
        model = clone(self.model)
        model, eval_history = self.model_adapter.fit(model, X_train_scaled, y_train, X_val_scaled, y_val, seed)
        return model, eval_history

    # ------------------------------------------------------------------
    # Predictions per fold
    # ------------------------------------------------------------------
    def _get_fold_predictions(
        self, model, X_val_scaled: np.ndarray, y_val: np.ndarray, selected_features: List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute predictions and probabilities for the fold."""
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = None
        if self.classes_to_save is not None and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val_scaled)
            if len(self.classes_to_save) == 1:
                y_pred_proba = proba[:, self.classes_to_save[0]]
            else:
                y_pred_proba = proba[:, self.classes_to_save]
        return y_pred, y_pred_proba

    # ------------------------------------------------------------------
    # Process a single fold
    # ------------------------------------------------------------------
    def process_fold(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, seed: int) -> FoldResult:
        """Run feature selection, balancing, scaling, training, and prediction for one fold."""
        
        np.random.seed(seed)
        random.seed(seed)
    
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

        # Apply feature selection
        X_train, X_val, selected_features = self._apply_feature_selection(X_train, y_train, X_val)

        # Apply balancing
        X_train, y_train = self._apply_balancer(X_train, y_train, selected_features, seed)

        # Apply scaling
        X_train_scaled, X_val_scaled, scaler = self._apply_scaler(X_train, y_train, X_val)

        # Train model + get eval history
        model, eval_history = self._train_model(X_train_scaled, y_train, X_val_scaled, y_val, seed)

        # Predictions
        y_pred, y_pred_proba = self._get_fold_predictions(model, X_val_scaled, y_val, selected_features)

        # Feature importances (if available)
        feature_importances = self.model_adapter.get_feature_importances(model, selected_features)

        return self.FoldResult(
            fold_idx, val_idx, selected_features, scaler, model, y_pred, y_pred_proba,
            feature_importances, eval_history
        )

    # ------------------------------------------------------------------
    # Parallel training
    # ------------------------------------------------------------------
    def parallel_training(self) -> List[FoldResult]:
        """Run all folds in parallel using joblib."""
        max_cores = cpu_count()
        print(f"[INFO] Maximum available cores: {max_cores}")

        if self.n_cores > max_cores:
            print(f"[WARNING] Requested n_cores={self.n_cores} exceeds available cores ({max_cores}). "
                  f"Using all {max_cores} cores instead.")
        
        n_jobs = self.n_cores if self.n_cores not in [0, None] else -1
        n_jobs = min(n_jobs, max_cores) if n_jobs > 0 else n_jobs

        n_splits = self.rkf.get_n_splits(self.X, self.y)
        seeds = self._generate_seeds(n_splits)

        results = Parallel(
            n_jobs=n_jobs,
            backend="loky"
        )(
            delayed(self.process_fold)(fold_idx, train_idx, val_idx, seeds[fold_idx])
            for fold_idx, (train_idx, val_idx) in tqdm(
                enumerate(self.rkf.split(self.X, self.y)), total=n_splits
            )
        )
        return results

    # ------------------------------------------------------------------
    # Output utilities
    # ------------------------------------------------------------------
    def get_feature_selection(self, results: List[FoldResult]) -> pd.DataFrame:
        """Return a DataFrame indicating which features were selected in each fold."""
        n_repeats = getattr(self.rkf, "n_repeats", 1)
        n_folds = int(self.rkf.get_n_splits(self.X, self.y) / n_repeats)

        row_names = [
            f"Iter_{r}_Fold_{f}"
            for r in range(1, n_repeats + 1)
            for f in range(1, n_folds + 1)
        ]

        df_feature_selection = pd.DataFrame(0, index=range(len(results)), columns=self.X.columns)
        for result in results:
            df_feature_selection.loc[result.fold_idx, result.selected_features] = 1

        df_feature_selection.index = row_names

        selected_counts = df_feature_selection.sum(axis=1).values
        print(
            f"Average selected features per fold: {selected_counts.mean():.2f} "
            f"(±{selected_counts.std():.2f})"
        )

        return df_feature_selection

    def get_predictions(self, results: List[FoldResult]) -> pd.DataFrame:
        """Return a DataFrame with fold predictions aligned to the original index."""
        predictions_dict = {idx: [] for idx in self.y.index}
        for result in results:
            for i, pos in enumerate(result.val_idx):
                real_idx = self.y.index[pos]
                predictions_dict[real_idx].append(result.y_pred[i])
        df_pred = pd.DataFrame.from_dict(predictions_dict, orient="index")
        df_pred.insert(0, "Label", self.y)
        df_pred.columns = ["Label"] + [f"Iter_{i+1}" for i in range(df_pred.shape[1] - 1)]
        df_pred = df_pred.sort_index()
        return df_pred

    def get_predictions_proba(self, results: List[FoldResult]) -> Optional[Union[pd.DataFrame, List[pd.DataFrame]]]:
        """Return probabilities for the selected classes across folds."""
        if self.classes_to_save is None:
            return None
        predictions_proba_dict = {
            class_idx: {idx: [] for idx in self.y.index} for class_idx in self.classes_to_save
        }
        for result in results:
            if result.y_pred_proba is None:
                continue
            for i, pos in enumerate(result.val_idx):
                real_idx = self.y.index[pos]
                if isinstance(result.y_pred_proba[i], (float, np.floating)):
                    predictions_proba_dict[self.classes_to_save[0]][real_idx].append(result.y_pred_proba[i])
                else:
                    for j, class_idx in enumerate(self.classes_to_save):
                        predictions_proba_dict[class_idx][real_idx].append(result.y_pred_proba[i][j])
        df_list = []
        for class_idx in self.classes_to_save:
            df_class = pd.DataFrame.from_dict(predictions_proba_dict[class_idx], orient="index")
            df_class.insert(0, "Label", self.y)
            df_class.columns = ["Label"] + [f"Iter_{i+1}" for i in range(df_class.shape[1] - 1)]
            df_class = df_class.sort_index()
            df_list.append(df_class)
        return df_list[0] if len(df_list) == 1 else df_list

    def get_feature_importances(self, results: List[FoldResult]) -> pd.DataFrame:
        """Aggregate feature importances across folds."""
        valid_results = [r for r in results if r.feature_importances is not None]
        if not valid_results:
            print("[INFO] No feature importances available.")
            return pd.DataFrame()
        df_importances = pd.DataFrame({r.fold_idx: r.feature_importances for r in valid_results}).T
        df_importances.index.name = "Fold"
        return df_importances

    def get_eval_history(self, results: List[FoldResult]) -> dict:
        """Return evaluation histories for models that support it (e.g., XGBoost)."""
        return {r.fold_idx: r.eval_history for r in results if r.eval_history is not None}

    def get_scaler_model(self, results: List[FoldResult]) -> List[Tuple[BaseEstimator, BaseEstimator]]:
        """Return the scaler and model objects for each fold."""
        return [(result.scaler, result.model) for result in results]

    def get_all(self, results: List[FoldResult]) -> dict:
        """Return all outputs: predictions, probabilities, feature selection, models, feature importances, and eval history."""
        return {
            "predictions": self.get_predictions(results),
            "predictions_proba": self.get_predictions_proba(results),
            "feature_selection": self.get_feature_selection(results),
            "scaler_model": self.get_scaler_model(results),
            "feature_importances": self.get_feature_importances(results),
            "eval_history": self.get_eval_history(results),
        }
