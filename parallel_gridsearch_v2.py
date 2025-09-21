# -*- coding: utf-8 -*-
"""
ParallelGridSearch - optimized for repeated cross-validation with parallel execution and grid search.
Author: mik16 (refactored with preprocessing cache, multi-metric aggregation, and modular design)
"""

#%% Libraries

import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from typing import List, Optional, Union, Tuple, Dict, Callable


#%% Machine learning

class ParallelGridSearch:
    """
    A utility class for parallelized repeated cross-validation combined with
    hyperparameter grid search. It supports optional feature selection,
    balancing, scaling, and model training with preprocessing cached per fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Union[pd.Series, np.ndarray]
        Target vector.
    rskf : object
        A cross-validation splitter (e.g., RepeatedKFold, RepeatedStratifiedKFold).
    scaler : BaseEstimator
        A scikit-learn compatible scaler (e.g., StandardScaler, MinMaxScaler).
    model : BaseEstimator
        A scikit-learn compatible model (e.g., LogisticRegression, XGBClassifier).
    param_grid : dict
        Dictionary defining the parameter grid for grid search.
    balancer : Optional[BaseEstimator], default=None
        A resampling strategy (e.g., SMOTE).
    feature_selectors : Optional[List[BaseEstimator]], default=None
        A list of feature selection transformers.
    classes_to_save : Optional[List[int]], default=None
        Class indices for which to save predicted probabilities.
    n_cores : int, default=-1
        Number of parallel jobs for joblib. -1 uses all available cores.
    parallel_over : str, {"fold", "param"}, default="fold"
        Defines whether to parallelize over folds or parameter combinations.
    save_models : bool, default=True
        If False, models are not stored in memory to save RAM.

    Attributes
    ----------
    FoldResult : namedtuple
        A container for storing fold-wise results:
        (fold_idx, val_idx, selected_features, scaler, param_combination,
         model, y_pred, y_pred_proba).
    """

    FoldResult = namedtuple(
        "FoldResult",
        [
            "fold_idx", "val_idx", "selected_features", "scaler",
            "param_combination", "model", "y_pred", "y_pred_proba"
        ],
    )

    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        rskf,
        scaler,
        model,
        param_grid: Dict,
        balancer=None,
        feature_selectors: Optional[List[BaseEstimator]] = None,
        classes_to_save: Optional[List[int]] = None,
        n_cores: int = -1,
        parallel_over: str = "fold",
        save_models: bool = True,
    ):
        self.X = X
        self.y = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        self.rskf = rskf
        self.scaler = scaler
        self.model = model
        self.balancer = balancer
        self.param_grid = list(ParameterGrid(param_grid))
        self.feature_selectors = feature_selectors or []
        self.classes_to_save = classes_to_save
        self.n_cores = n_cores
        self.parallel_over = parallel_over
        self.save_models = save_models

    # ------------------------------------------------------------------
    # Preprocessing fold-wise
    # ------------------------------------------------------------------
    def _apply_feature_selection(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Apply all feature selectors sequentially."""
        for selector in self.feature_selectors:
            if hasattr(selector, "set_output"):
                selector.set_output(transform="pandas")
            X_train = selector.fit_transform(X_train, y_train)
            X_val = selector.transform(X_val)
        return X_train, X_val, X_train.columns.tolist()

    def _apply_balancer(
        self, X_train: pd.DataFrame, y_train: pd.Series, selected_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply class balancing (e.g., SMOTE) if provided."""
        if self.balancer is not None:
            balancer = clone(self.balancer)
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
    # Preprocessing cache
    # ------------------------------------------------------------------
    def _preprocess_fold(self, fold_idx: int, train_idx, val_idx) -> dict:
        """Perform feature selection, balancing, and scaling ONCE per fold."""
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

        # Feature selection
        X_train, X_val, selected_features = self._apply_feature_selection(X_train, y_train, X_val)

        # Balancing
        X_train, y_train = self._apply_balancer(X_train, y_train, selected_features)

        # Scaling
        X_train_scaled, X_val_scaled, scaler = self._apply_scaler(X_train, y_train, X_val)

        return {
            "fold_idx": fold_idx,
            "val_idx": val_idx,
            "X_train_scaled": X_train_scaled,
            "X_val_scaled": X_val_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "selected_features": selected_features,
            "scaler": scaler,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_model(
        self, param_comb: Dict, X_train_scaled: np.ndarray, y_train: np.ndarray,
        X_val_scaled: np.ndarray, y_val: np.ndarray
    ) -> BaseEstimator:
        """Train the model with the provided parameter combination."""
        model = clone(self.model)
        model.set_params(**param_comb)
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        else:
            model.fit(X_train_scaled, y_train)
        return model

    def _get_fold_predictions(
        self, model, X_val_scaled: np.ndarray, y_val: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute predictions and probabilities for the fold."""
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = None
        if self.classes_to_save is not None:
            proba = model.predict_proba(X_val_scaled)
            if len(self.classes_to_save) == 1:
                y_pred_proba = proba[:, self.classes_to_save[0]]
            else:
                y_pred_proba = proba[:, self.classes_to_save]
        return y_pred, y_pred_proba

    def _train_on_preprocessed(self, preprocessed: dict, param_comb: dict):
        """Train a model using preprocessed fold data and a parameter combination."""
        model = self._train_model(
            param_comb,
            preprocessed["X_train_scaled"], preprocessed["y_train"],
            preprocessed["X_val_scaled"], preprocessed["y_val"]
        )

        # Predictions
        y_pred, y_pred_proba = self._get_fold_predictions(
            model, preprocessed["X_val_scaled"], preprocessed["y_val"]
        )

        return self.FoldResult(
            fold_idx=preprocessed["fold_idx"],
            val_idx=preprocessed["val_idx"],
            selected_features=preprocessed["selected_features"],
            scaler=preprocessed["scaler"],
            param_combination=param_comb,
            model=model if self.save_models else None,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )

    # ------------------------------------------------------------------
    # Parallel training
    # ------------------------------------------------------------------
    def parallel_training(self) -> List[FoldResult]:
        """Run folds and parameters in parallel with preprocessing cache."""
        max_cores = cpu_count()
        print(f"[INFO] Maximum available cores: {max_cores}")

        if self.n_cores > max_cores:
            print(
                f"[WARNING] Requested n_cores={self.n_cores} exceeds available cores ({max_cores}). "
                f"Using all {max_cores} cores instead."
            )

        n_splits = self.rskf.get_n_splits(self.X, self.y)

        # STEP 1: preprocess folds once
        print("[INFO] Preprocessing folds...")
        preprocessed_folds = [
            self._preprocess_fold(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in tqdm(
                enumerate(self.rskf.split(self.X, self.y)), total=n_splits
            )
        ]

        # STEP 2: parallel training
        if self.parallel_over == "fold":
            print("[INFO] Training parallelized over folds...")
            all_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                delayed(self._train_on_preprocessed)(preprocessed, param_comb)
                for preprocessed in tqdm(preprocessed_folds, desc="Training folds")
                for param_comb in self.param_grid
            )

        else:  # parallel_over == "param"
            print("[INFO] Training parallelized over parameters...")
            all_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                delayed(self._train_on_preprocessed)(preprocessed, param_comb)
                for param_comb in tqdm(self.param_grid, desc="Training params")
                for preprocessed in preprocessed_folds
            )

        return all_results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_results(
        self,
        results: List[FoldResult],
        metric_fn: Union[Callable, Dict[str, Callable]],
        use_proba: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregate results by computing mean and std of one or multiple metrics
        for each parameter set.

        Parameters
        ----------
        results : List[FoldResult]
            Results from parallel_training.
        metric_fn : callable or dict[str, callable]
            - Single metric function (e.g., recall_score, roc_auc_score), or
            - Dict of {metric_name: function}.
        use_proba : bool, default=False
            If True, use predicted probabilities instead of labels.
        show_progress : bool, default=True
            If True, display a tqdm progress bar.

        Returns
        -------
        df_summary : pd.DataFrame
            Summary DataFrame with mean and std for each metric,
            sorted by the first metric.
        """
        if callable(metric_fn):
            metrics = {"score": metric_fn}
        else:
            metrics = metric_fn

        metric_scores = defaultdict(lambda: defaultdict(list))
        iterator = tqdm(results, desc="Aggregating results") if show_progress else results

        for res in iterator:
            y_true = self.y.iloc[res.val_idx]
            y_pred, y_pred_proba = res.y_pred, res.y_pred_proba

            for name, fn in metrics.items():
                if use_proba and name=="auc":
                    if y_pred_proba is None:
                        continue
                    score = fn(y_true, y_pred_proba)
                else:
                    score = fn(y_true, y_pred)
                param_tuple = tuple(sorted(res.param_combination.items()))
                metric_scores[param_tuple][name].append(score)

        summary = []
        for param_tuple, scores_dict in metric_scores.items():
            param_dict = dict(param_tuple)
            for name, scores in scores_dict.items():
                param_dict[f"{name}_mean"] = np.mean(scores)
                param_dict[f"{name}_std"] = np.std(scores)
            summary.append(param_dict)

        df_summary = pd.DataFrame(summary)
        first_metric = list(metrics.keys())[0]
        df_summary = df_summary.sort_values(
            by=f"{first_metric}_mean", ascending=False
        ).reset_index(drop=True)

        return df_summary

    def get_best_params(self, df_summary: pd.DataFrame, metric: Optional[str] = None) -> Dict:
        """
        Return the best parameter set according to a chosen metric.

        Parameters
        ----------
        df_summary : pd.DataFrame
            Aggregated results from aggregate_results.
        metric : str, optional
            Metric name to use (e.g., "recall", "auc").
            If None, the first metric in df_summary is used.

        Returns
        -------
        best_params : dict
            Best parameter combination.
        """
        # Detect available metrics
        metric_names = [c.replace("_mean", "") for c in df_summary.columns if c.endswith("_mean")]
        if metric is None:
            metric = metric_names[0]
            print(f"[INFO] No metric specified, using '{metric}' as default.")

        if f"{metric}_mean" not in df_summary.columns:
            raise ValueError(f"Metric '{metric}' not found in df_summary. Available: {metric_names}")

        # Select best row by chosen metric
        best_row = df_summary.sort_values(by=f"{metric}_mean", ascending=False).iloc[0]

        # Drop metric columns to get only params
        exclude_cols = [c for c in df_summary.columns if c.endswith("_mean") or c.endswith("_std")]
        best_params = best_row.drop(exclude_cols).to_dict()

        print(f"Best Parameters (based on {metric}): {best_params}")
        return best_params

    def get_best_score(self, df_summary: pd.DataFrame, metric: Optional[str] = None) -> float:
        """
        Return the best mean score for a chosen metric.

        Parameters
        ----------
        df_summary : pd.DataFrame
            Aggregated results from aggregate_results.
        metric : str, optional
            Metric name to use (e.g., "recall", "auc").
            If None, the first metric in df_summary is used.

        Returns
        -------
        best_score : float
            The best mean score according to the chosen metric.
        """
        metric_names = [c.replace("_mean", "") for c in df_summary.columns if c.endswith("_mean")]
        if metric is None:
            metric = metric_names[0]
            print(f"[INFO] No metric specified, using '{metric}' as default.")

        if f"{metric}_mean" not in df_summary.columns:
            raise ValueError(f"Metric '{metric}' not found in df_summary. Available: {metric_names}")

        best_score = df_summary.sort_values(by=f"{metric}_mean", ascending=False).iloc[0][f"{metric}_mean"]

        print(f"Best {metric}: {best_score}")
        return best_score
