# -*- coding: utf-8 -*-
"""
ParallelGridSearch - optimized for repeated cross-validation with parallel execution and grid search.
Author: mik16 
"""

import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed, cpu_count
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm
from typing import List, Optional, Union, Dict, Callable


class ParallelGridSearch:
    """
    Parallelized repeated cross-validation with hyperparameter grid search.
    Supports sklearn/imbalanced-learn pipelines as estimator.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Union[pd.Series, np.ndarray]
        Target vector.
    rskf : object
        A cross-validation splitter (e.g., RepeatedKFold, RepeatedStratifiedKFold).
    estimator : BaseEstimator
        A scikit-learn compatible estimator (e.g., Pipeline with preprocessing and model).
    param_grid : dict
        Dictionary defining the parameter grid for grid search.
    classes_to_save : Optional[List[int]], default=None
        Class indices for which to save predicted probabilities.
    n_cores : int, default=-1
        Number of parallel jobs for joblib. -1 uses all available cores.
    parallel_over : str, {"fold", "param"}, default="fold"
        Defines whether to parallelize over folds or parameter combinations.
    save : list of {"model", "scaler", "features"}, optional
        Which objects to save in FoldResult. Default=["model"].
    """

    FoldResult = namedtuple(
        "FoldResult",
        [
            "fold_idx", "val_idx", "param_combination",
            "model", "scaler", "selected_features",
            "y_pred", "y_pred_proba"
        ],
    )

    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        rskf,
        estimator: BaseEstimator,
        param_grid: Dict,
        classes_to_save: Optional[List[int]] = None,
        n_cores: int = -1,
        parallel_over: str = "fold",
        save: Optional[List[str]] = None,
    ):
        self.X = X
        self.y = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        self.rskf = rskf
        self.estimator = estimator
        self.param_grid = list(ParameterGrid(param_grid))
        self.classes_to_save = classes_to_save
        self.n_cores = n_cores
        self.parallel_over = parallel_over
        self.save = save or ["model"]

    # ------------------------------------------------------------------
    # Training + Prediction
    # ------------------------------------------------------------------
    def _train_and_evaluate(self, fold_idx, train_idx, val_idx, param_comb):
        """Train estimator with given params on a fold and return predictions."""
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
    
        # Clone estimator and set params
        model = clone(self.estimator)
        model.set_params(**param_comb)
        
        # Fit parameters for XGB using set_params
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            final_est = model.named_steps["model"]
        
            # Special case: XGB
            if isinstance(final_est, (XGBClassifier, XGBRegressor)):
                # Partial fit of the pipeline (only preprocessing and feature selection)
                model[:-1].fit(X_train, y_train)
        
                # Transform X_val using the same transformations as X_train
                X_val_trans = model[:-1].transform(X_val)
        
                # Pass eval_set and verbose to the 'model' step via set_params
                model.set_params(model__eval_set=[(X_val_trans, y_val)],
                                 model__verbose=False)
        
                # Full fit of the pipeline including the XGB model
                model.fit(X_train, y_train)
            else:
                # Regular fit for non-XGB models
                model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = None
        if self.classes_to_save is not None and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            if len(self.classes_to_save) == 1:
                y_pred_proba = proba[:, self.classes_to_save[0]]
            else:
                y_pred_proba = proba[:, self.classes_to_save]
    
        # Optional saves
        saved_model = model if "model" in self.save else None
        saved_scaler, saved_features = None, None
    
        if "scaler" in self.save and "scaler" in model.named_steps:
            saved_scaler = model.named_steps["scaler"]
    
        if "features" in self.save:
            # try to reconstruct feature names after all selection steps
            current_features = self.X.columns
            for step_name, step in model.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    current_features = step.get_feature_names_out(current_features)
                elif hasattr(step, "get_support"):
                    mask = step.get_support()
                    current_features = current_features[mask]
            saved_features = list(current_features)
    
        return self.FoldResult(
            fold_idx=fold_idx,
            val_idx=val_idx,
            param_combination=param_comb,
            model=saved_model,
            scaler=saved_scaler,
            selected_features=saved_features,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )

    # ------------------------------------------------------------------
    # Parallel training
    # ------------------------------------------------------------------
    def parallel_training(self) -> List[FoldResult]:
        """Run all folds and parameter combinations in parallel."""
        max_cores = cpu_count()
        print(f"[INFO] Maximum available cores: {max_cores}")

        if self.n_cores > max_cores:
            print(
                f"[WARNING] Requested n_cores={self.n_cores} exceeds available cores ({max_cores}). "
                f"Using all {max_cores} cores instead."
            )

        n_splits = self.rskf.get_n_splits(self.X, self.y)

        if self.parallel_over == "fold":
            print("[INFO] Training parallelized over folds...")
            all_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                delayed(self._train_and_evaluate)(fold_idx, train_idx, val_idx, param_comb)
                for fold_idx, (train_idx, val_idx) in tqdm(
                    enumerate(self.rskf.split(self.X, self.y)), total=n_splits, desc="Folds"
                )
                for param_comb in self.param_grid
            )
        else:  # parallel_over == "param"
            print("[INFO] Training parallelized over parameters...")
            all_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                delayed(self._train_and_evaluate)(fold_idx, train_idx, val_idx, param_comb)
                for param_comb in tqdm(self.param_grid, desc="Params")
                for fold_idx, (train_idx, val_idx) in enumerate(self.rskf.split(self.X, self.y))
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
        """Return best parameter set according to chosen metric."""
        metric_names = [c.replace("_mean", "") for c in df_summary.columns if c.endswith("_mean")]
        if metric is None:
            metric = metric_names[0]
            print(f"[INFO] No metric specified, using '{metric}' as default.")

        best_row = df_summary.sort_values(by=f"{metric}_mean", ascending=False).iloc[0]
        exclude_cols = [c for c in df_summary.columns if c.endswith("_mean") or c.endswith("_std")]
        return best_row.drop(exclude_cols).to_dict()

    def get_best_score(self, df_summary: pd.DataFrame, metric: Optional[str] = None) -> float:
        """Return best mean score for chosen metric."""
        metric_names = [c.replace("_mean", "") for c in df_summary.columns if c.endswith("_mean")]
        if metric is None:
            metric = metric_names[0]
            print(f"[INFO] No metric specified, using '{metric}' as default.")
        return df_summary.sort_values(by=f"{metric}_mean", ascending=False).iloc[0][f"{metric}_mean"]
