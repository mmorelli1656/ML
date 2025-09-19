# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:03:30 2025

@author: mik16
"""

#%% Libreries

import shap
import pandas as pd
from typing import List, Dict, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


#%% XAI with SHAP

class SHAPHandler:
    """
    A class for computing and aggregating SHAP values across repeated
    cross-validation folds with optional parallelization.

    Parameters
    ----------
    results : List
        A list of fold results containing trained models, validation indices,
        selected features, and optionally scalers.
    X : pd.DataFrame
        The original dataset (features only).
    rkf : object
        A repeated cross-validation splitter (e.g., RepeatedKFold).
    explainer_type : {"auto", "tree", "linear", "kernel"}, default="auto"
        Type of SHAP explainer to use:
        - "auto": try TreeExplainer, otherwise fallback to generic Explainer
        - "tree": TreeExplainer
        - "linear": LinearExplainer
        - "kernel": KernelExplainer
    use_scaled : bool, default=False
        Whether to use the scaled features (if a scaler is available in results).
    n_jobs : int, default=1
        Number of jobs for parallel execution (joblib).
    backend : str, default="loky"
        Backend for joblib parallelization.
    parallel_level : {"fold", "repeat", None}, optional
        Level of parallelization:
        - "fold": parallelize over individual folds
        - "repeat": parallelize over repetitions
        - None: sequential execution

    Attributes
    ----------
    shap_dict_ : dict of {int: pd.DataFrame}
        Dictionary where keys are repetition indices (1..n_repeats),
        and values are DataFrames of SHAP values aggregated across folds.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.model_selection import RepeatedKFold
    >>>
    >>> # Load dataset
    >>> X, y = load_boston(return_X_y=True, as_frame=True)
    >>> rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    >>>
    >>> # Mock results object (usually created during CV)
    >>> class FoldResult:
    ...     def __init__(self, model, fold_idx, val_idx, selected_features, scaler=None):
    ...         self.model = model
    ...         self.fold_idx = fold_idx
    ...         self.val_idx = val_idx
    ...         self.selected_features = selected_features
    ...         self.scaler = scaler
    >>>
    >>> results = []
    >>> for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(X, y)):
    ...     model = RandomForestRegressor().fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     res = FoldResult(model, fold_idx, val_idx, list(X.columns))
    ...     results.append(res)
    >>>
    >>> handler = SHAPHandler(results, X, rkf, explainer_type="tree")
    >>> shap_dict = handler.compute_shap_values()
    >>> top_features = handler.plot_summary_aggregated(max_display=10)
    """

    def __init__(
        self,
        results: List,
        X: pd.DataFrame,
        rkf,
        explainer_type: str = "auto",
        use_scaled: bool = False,
        n_jobs: int = 1,
        backend: str = "loky",
        parallel_level: Optional[str] = None,  # "fold", "repeat", None
    ):
        self.results = results
        self.X = X
        self.rkf = rkf
        self.explainer_type = explainer_type
        self.use_scaled = use_scaled
        self.n_jobs = n_jobs
        self.backend = backend
        self.parallel_level = parallel_level

        self.shap_dict_: Optional[Dict[int, pd.DataFrame]] = None

    def _get_explainer(self, model, result):
        """
        Return an appropriate SHAP explainer based on the selected type.
        """
        if self.explainer_type == "auto":
            try:
                return shap.TreeExplainer(model)
            except Exception:
                return shap.Explainer(model)
    
        elif self.explainer_type == "tree":
            return shap.TreeExplainer(model)
    
        elif self.explainer_type == "linear":
            train_idx = np.setdiff1d(np.arange(len(self.X)), result.val_idx)
            background = self.X.iloc[train_idx][result.selected_features]
            if self.use_scaled and result.scaler is not None:
                background = result.scaler.transform(background)
        
            try:
                return shap.LinearExplainer(
                    model.predict_proba,
                    background,
                    link=shap.links.identity  # più robusto
                )
            except TypeError:
                # fallback per versioni che accettano solo stringhe
                return shap.LinearExplainer(
                    model.predict_proba,
                    background,
                    link="identity"
                )
        
        elif self.explainer_type == "kernel":
            train_idx = np.setdiff1d(np.arange(len(self.X)), result.val_idx)
            background = self.X.iloc[train_idx][result.selected_features]
            
            if self.use_scaled and result.scaler is not None:
                background = result.scaler.transform(background)
            return shap.KernelExplainer(
                model.predict_proba,  # <--- usa proba, non predict
                background
            )

        else:
            raise ValueError(f"Explainer type '{self.explainer_type}' is not supported.")

    def _compute_fold_shap(self, result, n_folds):
        """
        Compute SHAP values for a single fold.
    
        Parameters
        ----------
        result : object
            A single fold result containing model, validation indices,
            selected features, and optional scaler.
        n_folds : int
            Number of folds per repetition.
    
        Returns
        -------
        repeat_idx : int
            Index of the repetition this fold belongs to.
        df_shap : pd.DataFrame
            DataFrame with SHAP values for the validation set,
            aligned with the original feature space (non-selected features are NaN).
        """
        # Identify repetition index based on fold index
        repeat_idx = (result.fold_idx // n_folds) + 1
        val_idx = result.val_idx
    
        # Restrict validation data to selected features only
        X_val = self.X.iloc[val_idx][result.selected_features]
    
        # Optionally scale validation data (if scaler is provided in results)
        if self.use_scaled and result.scaler is not None:
            X_val = result.scaler.transform(X_val)
    
        # Build SHAP explainer for the current model
        explainer = self._get_explainer(result.model, result)
    
        # Compute SHAP values for the validation set
        shap_values = explainer(X_val)
    
        # --- Robust handling for different SHAP outputs ---
        if isinstance(shap_values, list):
            # Old API returns a list of arrays, one per class
            # For binary classification, take the positive class (index 1)
            if len(shap_values) == 2:
                shap_values_2d = shap_values[1]
            else:
                shap_values_2d = shap_values[0]  # single class / regression
        elif isinstance(shap_values.values, np.ndarray):
            if shap_values.values.ndim == 3 and shap_values.values.shape[2] == 2:
                # shape = (n_samples, n_features, n_classes)
                shap_values_2d = shap_values.values[:, :, 1]  # take positive class
            else:
                shap_values_2d = shap_values.values
        else:
            raise ValueError("Unsupported SHAP output type.")
    
        # Create a DataFrame aligned with original dataset
        df_shap = pd.DataFrame(
            index=self.X.index[val_idx],
            columns=self.X.columns,
            data=pd.NA
        )
        df_shap[result.selected_features] = shap_values_2d
    
        return repeat_idx, df_shap


    def compute_shap_values(self) -> Dict[int, pd.DataFrame]:
        """
        Compute SHAP values for all folds and aggregate them by repetition.

        Returns
        -------
        shap_dict : dict of {int: pd.DataFrame}
            Dictionary mapping repetition indices to DataFrames of SHAP values.
        """
        # Get number of repetitions and folds per repetition
        n_repeats = getattr(self.rkf, "n_repeats", 1)
        n_folds = int(self.rkf.get_n_splits(self.X) / n_repeats)

        # Initialize dict for results
        shap_dict = {r: [] for r in range(1, n_repeats + 1)}

        if self.parallel_level == "fold":
            # Parallelize across folds
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(self._compute_fold_shap)(res, n_folds)
                for res in tqdm(self.results, desc="Computing SHAP per fold")
            )
            # Collect results
            for repeat_idx, df_shap in results:
                shap_dict[repeat_idx].append(df_shap)

        elif self.parallel_level == "repeat":
            # Parallelize across repetitions
            def process_repeat(r):
                folds = []
                # Process only folds belonging to repetition r
                for res in [res for res in self.results if (res.fold_idx // n_folds) + 1 == r]:
                    _, df_shap = self._compute_fold_shap(res, n_folds)
                    folds.append(df_shap)
                return r, folds

            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(process_repeat)(r)
                for r in tqdm(range(1, n_repeats + 1), desc="Computing SHAP per repetition")
            )
            for r, folds in results:
                shap_dict[r].extend(folds)

        else:
            # Sequential execution (no parallelization)
            for res in tqdm(self.results, desc="Computing SHAP sequentially"):
                repeat_idx, df_shap = self._compute_fold_shap(res, n_folds)
                shap_dict[repeat_idx].append(df_shap)

        # Aggregate folds for each repetition into a single DataFrame
        for r in shap_dict:
            # Drop columns that are entirely NaN (not selected in any fold)
            folds = [df.dropna(axis=1, how="all") for df in shap_dict[r]]
            # Concatenate fold results along rows and order by sample index
            shap_dict[r] = pd.concat(folds).sort_index()

        self.shap_dict_ = shap_dict
        return shap_dict

    def plot_summary_aggregated(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot an aggregated SHAP summary across all repetitions and folds.
    
        This method concatenates SHAP values from all repetitions and folds,
        fills NaNs with 0 (features not selected in some folds), and plots
        the global summary plot using SHAP.
    
        Parameters
        ----------
        max_display : int, default=20
            Maximum number of features to display in the plot.
        what : str, default="Target"
            Label for the plot title.
        save_path : str, optional
            If provided, saves the plot to the specified path.
    
        Returns
        -------
        top_features : pd.DataFrame
            DataFrame containing the top features ranked by mean absolute SHAP value.
        """
        # Ensure SHAP values have been computed
        if self.shap_dict_ is None:
            raise RuntimeError("You must call compute_shap_values() before plotting!")
    
        # Concatenate all repetitions
        df_shap_all = pd.concat([self.shap_dict_[r] for r in self.shap_dict_], axis=0)
    
        # Use original feature values for plotting
        df_features_all = self.X.loc[df_shap_all.index, df_shap_all.columns]
    
        # Convert NaN to 0 for features not selected in some folds
        shap_values_concatenated = df_shap_all.fillna(0).values
        features_values = df_features_all.fillna(0).values
    
        # Plot summary
        plt.title("SHAP Summary Plot - Global case", fontsize=15, loc='center')
        shap.summary_plot(
            shap_values_concatenated,
            features_values,
            feature_names=df_shap_all.columns,
            max_display=max_display,
            show=False
        )
        if save_path is not None:
            file_path = Path(save_path) / "shap_summary.png"
            plt.savefig(file_path, bbox_inches='tight', dpi=200)
        plt.show()
    
        # Compute mean absolute SHAP values for ranking
        mean_abs_shap = np.mean(np.abs(shap_values_concatenated), axis=0)
        feature_importance = pd.DataFrame(
            {"MeanAbsSHAP": mean_abs_shap},
            index=df_shap_all.columns
        )
        top_features = feature_importance.sort_values(
            by="MeanAbsSHAP", ascending=False
        ).head(max_display)
    
        return top_features
