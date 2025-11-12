# -*- coding: utf-8 -*-
"""
ParallelModelTrainer - optimized for repeated cross-validation with parallel execution.
Author: mik16
"""

import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.base import clone, BaseEstimator
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from typing import List, Optional, Union, Tuple


class ParallelModelTrainer:
    """
    A utility class for parallelized repeated cross-validation with optional
    feature selection, balancing, scaling, and model training.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Union[pd.Series, np.ndarray]
        Target vector.
    rkf : object
        A cross-validation splitter (e.g., RepeatedKFold, RepeatedStratifiedKFold).
    scaler : BaseEstimator
        A scikit-learn compatible scaler (e.g., StandardScaler, MinMaxScaler).
    model : BaseEstimator
        A scikit-learn compatible model (e.g., LogisticRegression, XGBClassifier).
    balancer : Optional[BaseEstimator], default=None
        A resampling strategy (e.g., SMOTE).
    feature_selectors : Optional[List[BaseEstimator]], default=None
        A list of feature selection transformers.
    classes_to_save : Optional[List[int]], default=None
        Class indices for which to save predicted probabilities.
    n_cores : int, default=-1
        Number of parallel jobs for joblib. -1 uses all available cores.

    Attributes
    ----------
    FoldResult : namedtuple
        A container for storing fold-wise results:
        (fold_idx, val_idx, selected_features, scaler, model, y_pred, y_pred_proba).
    """

    FoldResult = namedtuple(
        "FoldResult",
        ["fold_idx", "val_idx", "selected_features", "scaler", "model", "y_pred", "y_pred_proba"],
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
    # Training fold-wise
    # ------------------------------------------------------------------
    def _train_model(
        self, X_train_scaled: np.ndarray, y_train: np.ndarray,
        X_val_scaled: np.ndarray, y_val: np.ndarray
    ) -> BaseEstimator:
        """Train the model. Special handling for XGBoost models with eval_set."""
        model = clone(self.model)
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        else:
            model.fit(X_train_scaled, y_train)
        return model

    # ------------------------------------------------------------------
    # Predictions fold-wise
    # ------------------------------------------------------------------
    def _get_fold_predictions(
        self, model, X_val_scaled: np.ndarray, y_val: np.ndarray, selected_features: List[str]
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

    # ------------------------------------------------------------------
    # Process a single fold
    # ------------------------------------------------------------------
    def process_fold(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> FoldResult:
        """Run feature selection, balancing, scaling, training, and prediction for one fold."""
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

        # Apply feature selection
        X_train, X_val, selected_features = self._apply_feature_selection(X_train, y_train, X_val)

        # Apply balancing
        X_train, y_train = self._apply_balancer(X_train, y_train, selected_features)

        # Apply scaling
        X_train_scaled, X_val_scaled, scaler = self._apply_scaler(X_train, y_train, X_val)

        # Train model
        model = self._train_model(X_train_scaled, y_train, X_val_scaled, y_val)

        # Predictions
        y_pred, y_pred_proba = self._get_fold_predictions(model, X_val_scaled, y_val, selected_features)

        return self.FoldResult(fold_idx, val_idx, selected_features, scaler, model, y_pred, y_pred_proba)

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
    
        n_splits = self.rkf.get_n_splits(self.X, self.y)
    
        results = Parallel(
            n_jobs=self.n_cores,  # se > max_cores, joblib lo limita automaticamente
            backend="loky"
        )(
            delayed(self.process_fold)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in tqdm(
                enumerate(self.rkf.split(self.X, self.y)), total=n_splits
            )
        )
        return results

    # ------------------------------------------------------------------
    # Output utilities
    # ------------------------------------------------------------------
    def get_feature_selection(self, results: List[FoldResult]) -> pd.DataFrame:
        """Return a DataFrame indicating which features were selected in each fold.
        
        Also prints the average number of selected features across folds
        with its standard deviation.
        """
        # Numero di ripetizioni e fold
        n_repeats = getattr(self.rkf, "n_repeats", 1)
        n_folds = int(self.rkf.get_n_splits(self.X, self.y) / n_repeats)
    
        # Nomi leggibili per i fold
        row_names = [
            f"Iter_{r}_Fold_{f}"
            for r in range(1, n_repeats + 1)
            for f in range(1, n_folds + 1)
        ]
    
        # DataFrame vuoto con indice numerico coerente con result.fold_idx
        df_feature_selection = pd.DataFrame(
            0, index=range(len(results)), columns=self.X.columns
        )
    
        # Riempimento: 1 se la feature è selezionata
        for result in results:
            df_feature_selection.loc[result.fold_idx, result.selected_features] = 1
    
        # Sostituisci l’indice numerico con i nomi leggibili
        df_feature_selection.index = row_names
    
        # Statistiche
        selected_counts = df_feature_selection.sum(axis=1).values
        mean_selected = selected_counts.mean()
        std_selected = selected_counts.std()
    
        print(
            f"Media feature selezionate per fold: {mean_selected:.2f} "
            f"(±{std_selected:.2f})"
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

    def get_scaler_model(self, results: List[FoldResult]) -> List[Tuple[BaseEstimator, BaseEstimator]]:
        """Return the scaler and model objects for each fold."""
        return [(result.scaler, result.model) for result in results]

    def get_all(self, results: List[FoldResult]) -> dict:
        """Return all outputs: predictions, probabilities, feature selection, and models."""
        return {
            "predictions": self.get_predictions(results),
            "predictions_proba": self.get_predictions_proba(results),
            "feature_selection": self.get_feature_selection(results),
            "scaler_model": self.get_scaler_model(results),
        }
