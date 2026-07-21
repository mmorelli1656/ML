# -*- coding: utf-8 -*-
"""
eXplainability with SHAP - computes shapley values with a selected explainer.
@author: mik16
"""

import shap
import pandas as pd
from typing import List, Dict, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class SHAPHandler:
    """
    A class for computing and aggregating SHAP values across repeated
    cross-validation folds with optional parallelization.

    The constructor only holds *configuration* (how SHAP values should be
    computed). The actual data (results, X, rkf) is passed to
    `compute_shap_values()`, so the same configured handler can be reused
    across different result sets without re-instantiation.

    Parameters
    ----------
    explainer_type : {"auto", "tree", "linear", "kernel"}, default="auto"
        Type of SHAP explainer to use:
        - "auto": try TreeExplainer, otherwise fallback to generic Explainer
        - "tree": TreeExplainer
        - "linear": LinearExplainer
        - "kernel": KernelExplainer
    n_jobs : int, default=1
        Number of jobs for parallel execution (joblib).
    backend : str, default="loky"
        Backend for joblib parallelization.
    parallel_level : {"fold", "repeat", None}, optional
        Level of parallelization:
        - "fold": parallelize over individual folds
        - "repeat": parallelize over repetitions
        - None: sequential execution (default; no joblib/loky involved)

    Attributes
    ----------
    shap_dict_ : dict of {int: pd.DataFrame}
        Populated after calling `compute_shap_values()`. Keys are repetition
        indices (1..n_repeats), values are DataFrames of SHAP values
        aggregated across folds.
    X_ : pd.DataFrame
        The dataset used in the last call to `compute_shap_values()`.
        Stored so that `SHAPPlotter` can reuse it without needing it passed
        in again.
    target_class_ : int or label or None
        The class selected in the last call to `compute_shap_values()`.

    Examples
    --------
    >>> handler = SHAPHandler(explainer_type="tree", n_jobs=4, parallel_level="fold")
    >>> shap_dict = handler.compute_shap_values(results, X, rkf, use_scaled=True,
    ...                                          target_class=1)
    >>>
    >>> plotter = SHAPPlotter(handler)
    >>> top_features = plotter.plot_summary_aggregated(max_display=10)
    """

    def __init__(
        self,
        explainer_type: str = "auto",
        n_jobs: int = 1,
        backend: str = "loky",
        parallel_level: Optional[str] = None,  # "fold", "repeat", None
    ):
        self.explainer_type = explainer_type
        self.n_jobs = n_jobs
        self.backend = backend
        self.parallel_level = parallel_level

        # Populated by compute_shap_values()
        self.shap_dict_: Optional[Dict[int, pd.DataFrame]] = None
        self.X_: Optional[pd.DataFrame] = None
        self.target_class_ = None

    def _get_explainer(self, result, X, use_scaled):
        """
        Return an appropriate SHAP explainer based on the selected type.
        """
        if self.explainer_type == "auto":
            try:
                return shap.TreeExplainer(result.model)
            except Exception:
                return shap.Explainer(result.model)

        elif self.explainer_type == "tree":
            return shap.TreeExplainer(result.model)

        elif self.explainer_type == "linear":
            train_idx = np.setdiff1d(np.arange(len(X)), result.val_idx)
            background = X.iloc[train_idx][result.selected_features]
            if use_scaled and result.scaler is not None:
                background = result.scaler.transform(background)

            return shap.LinearExplainer(
                result.model,
                background)

        elif self.explainer_type == "kernel":
            train_idx = np.setdiff1d(np.arange(len(X)), result.val_idx)
            background = X.iloc[train_idx][result.selected_features]

            if use_scaled and result.scaler is not None:
                background = result.scaler.transform(background)
            return shap.KernelExplainer(
                result.model.predict_proba,
                background
            )

        else:
            raise ValueError(f"Explainer type '{self.explainer_type}' is not supported.")

    def _class_to_index(self, classes, target_class, n_classes):
        """
        Map `target_class` (a real class label, e.g. 1 or "yes") to its
        positional index in the SHAP output, using `model.classes_` when
        available. Falls back to treating `target_class` as a positional
        index if the model exposes no `.classes_` attribute.
        """
        if classes is not None:
            classes = list(classes)
            if target_class not in classes:
                raise ValueError(
                    f"target_class={target_class!r} not found in "
                    f"model.classes_={classes}."
                )
            return classes.index(target_class)

        if not (0 <= target_class < n_classes):
            raise ValueError(
                f"target_class={target_class} out of range for {n_classes} "
                "classes (model has no `.classes_` attribute, so target_class "
                "is treated as a positional index)."
            )
        return target_class

    def _select_class_slice(self, shap_values, model, target_class=None):
        """
        Select the 2D (n_samples, n_features) SHAP slice for a single class,
        handling both binary and multiclass outputs, and both the old
        (list-of-arrays) and current (Explanation with 3D .values) SHAP APIs.

        Binary models (2 classes): `target_class` is OPTIONAL. Defaults to
        the positive class (index 1 of `model.classes_`) if not given,
        matching the previous implicit behaviour. Pass `target_class`
        explicitly to select the other class instead.

        Multiclass models (3+ classes): `target_class` is REQUIRED. There
        is no sensible default, since aggregating classes together would
        mix signs and mislead any downstream ranking. Omitting it raises a
        descriptive ValueError instead of silently picking a class.

        Parameters
        ----------
        shap_values : list or shap.Explanation
            Raw output of `explainer(X_val)`.
        model : object
            The fold's trained model (used to read `.classes_` if
            available, so `target_class` can be given as an actual class
            label rather than a positional index).
        target_class : int or label, optional
            Which class's SHAP values to select.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        classes = getattr(model, "classes_", None)

        # --- Old API: list of arrays, one per class ---
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
            if n_classes == 2:
                if target_class is None:
                    return shap_values[1]  # positive class by convention
                idx = self._class_to_index(classes, target_class, n_classes)
                return shap_values[idx]
            if target_class is None:
                raise ValueError(
                    f"Model has {n_classes} classes; SHAP values are computed "
                    "per-class. You must specify `target_class` explicitly "
                    "(e.g. target_class=1) to select which class to use."
                )
            idx = self._class_to_index(classes, target_class, n_classes)
            return shap_values[idx]

        # --- Current API: Explanation object with .values ---
        values = shap_values.values if hasattr(shap_values, "values") else shap_values
        if not isinstance(values, np.ndarray):
            raise ValueError("Unsupported SHAP output type.")

        if values.ndim == 2:
            # Already (n_samples, n_features): regression or single-output binary
            return values

        if values.ndim == 3:
            n_classes = values.shape[2]
            if n_classes == 2:
                if target_class is None:
                    return values[:, :, 1]
                idx = self._class_to_index(classes, target_class, n_classes)
                return values[:, :, idx]
            if target_class is None:
                raise ValueError(
                    f"Model has {n_classes} classes; SHAP values have shape "
                    f"{values.shape}. You must specify `target_class` explicitly "
                    "(e.g. target_class=1) to select which class to use."
                )
            idx = self._class_to_index(classes, target_class, n_classes)
            return values[:, :, idx]

        raise ValueError(f"Unexpected SHAP values array shape: {values.shape}")

    def _compute_fold_shap(self, result, n_folds, X, use_scaled, target_class=None):
        """
        Compute SHAP values for a single fold.

        Parameters
        ----------
        result : object
            A single fold result containing model, validation indices,
            selected features, and optional scaler.
        n_folds : int
            Number of folds per repetition.
        X : pd.DataFrame
            The dataset for this computation.
        use_scaled : bool
            Whether to use the scaled features (if a scaler is available
            in `result`).
        target_class : int or label, optional
            Which class's SHAP values to compute. Optional for binary
            models (defaults to the positive class), REQUIRED for
            multiclass models (3+ classes).

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
        X_val = X.iloc[val_idx][result.selected_features]

        # Optionally scale validation data (if scaler is provided in results)
        if use_scaled and result.scaler is not None:
            X_val = result.scaler.transform(X_val)

        # Build SHAP explainer for the current model
        explainer = self._get_explainer(result, X, use_scaled)

        # Compute SHAP values for the validation set
        shap_values = explainer(X_val)

        # Select the correct class slice (binary: optional target_class,
        # multiclass: required target_class) instead of silently guessing
        shap_values_2d = self._select_class_slice(shap_values, result.model, target_class)

        # Create a DataFrame aligned with original dataset
        df_shap = pd.DataFrame(
            index=X.index[val_idx],
            columns=X.columns,
            data=pd.NA
        )
        df_shap[result.selected_features] = shap_values_2d

        return repeat_idx, df_shap

    def compute_shap_values(
        self,
        results: List,
        X: pd.DataFrame,
        rkf,
        use_scaled: bool = False,
        target_class=None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Compute SHAP values for all folds and aggregate them by repetition.

        This is the single entry point for a computation: it takes the data
        (results, X, rkf), computes everything, and stores internal state
        (`self.X_`, `self.shap_dict_`) so that `SHAPPlotter` can plot
        afterwards without needing the data passed again.

        Parameters
        ----------
        results : List
            A list of fold results containing trained models, validation
            indices, selected features, and optionally scalers.
        X : pd.DataFrame
            The original dataset (features only).
        rkf : object
            A repeated cross-validation splitter (e.g., RepeatedKFold).
        use_scaled : bool, default=False
            Whether to use the scaled features for this computation
            (if a scaler is available in `results`). Tied to this specific
            (results, X) pair rather than to the handler's configuration,
            since whether scaling applies depends on how `results` was built.
        target_class : int or label, optional
            Which class's SHAP values to compute.
            - Binary models (2 classes): optional. Defaults to the
              positive class (index 1 of `model.classes_`) if omitted.
            - Multiclass models (3+ classes): REQUIRED. Omitting it raises
              a descriptive ValueError rather than silently picking a
              class, since aggregating classes together would mix signs
              and mislead any downstream ranking.
            The value is the actual class label (e.g. 1, "yes"), not a
            positional index; it's resolved against `model.classes_`.

        Returns
        -------
        shap_dict : dict of {int: pd.DataFrame}
            Dictionary mapping repetition indices to DataFrames of SHAP values,
            for the selected `target_class`.
        """
        # Get number of repetitions and folds per repetition
        n_repeats = getattr(rkf, "n_repeats", 1)
        n_folds = int(rkf.get_n_splits(X) / n_repeats)

        # Initialize dict for results
        shap_dict = {r: [] for r in range(1, n_repeats + 1)}

        if self.parallel_level == "fold":
            # Parallelize across folds
            fold_results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(self._compute_fold_shap)(res, n_folds, X, use_scaled, target_class)
                for res in tqdm(results, desc="Computing SHAP per fold")
            )
            # Collect results
            for repeat_idx, df_shap in fold_results:
                shap_dict[repeat_idx].append(df_shap)

        elif self.parallel_level == "repeat":
            # Parallelize across repetitions
            def process_repeat(r):
                folds = []
                # Process only folds belonging to repetition r
                for res in [res for res in results if (res.fold_idx // n_folds) + 1 == r]:
                    _, df_shap = self._compute_fold_shap(res, n_folds, X, use_scaled, target_class)
                    folds.append(df_shap)
                return r, folds

            repeat_results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(process_repeat)(r)
                for r in tqdm(range(1, n_repeats + 1), desc="Computing SHAP per repetition")
            )
            for r, folds in repeat_results:
                shap_dict[r].extend(folds)

        else:
            # Sequential execution (no parallelization)
            for res in tqdm(results, desc="Computing SHAP sequentially"):
                repeat_idx, df_shap = self._compute_fold_shap(res, n_folds, X, use_scaled, target_class)
                shap_dict[repeat_idx].append(df_shap)

        # Aggregate folds for each repetition into a single DataFrame
        for r in shap_dict:
            # Drop columns that are entirely NaN (not selected in any fold)
            folds = [df.dropna(axis=1, how="all") for df in shap_dict[r]]
            # Concatenate fold results along rows and order by sample index
            shap_dict[r] = pd.concat(folds).sort_index()

        self.shap_dict_ = shap_dict
        self.X_ = X
        self.target_class_ = target_class
        return shap_dict


class SHAPPlotter:
    """
    Visualization for SHAP values computed by `SHAPHandler`.

    Mirrors the EvaluationMetrics / MetricsPlotter split: the plotter reads
    state (`shap_dict_`, `X_`) off an already-computed `SHAPHandler` instance,
    so plotting never needs the raw data passed in again.

    Parameters
    ----------
    handler : SHAPHandler
        A handler instance on which `compute_shap_values()` has already
        been called.
    show : bool, default=True
        Default value for whether plots call `plt.show()`. Set to False
        if running headless / only saving figures (e.g. some Spyder setups
        or batch scripts), and override per-call via the `show` argument.

    Examples
    --------
    >>> handler = SHAPHandler(explainer_type="tree")
    >>> handler.compute_shap_values(results, X, rkf, target_class=1)
    >>> plotter = SHAPPlotter(handler)
    >>> top_features = plotter.plot_summary_aggregated(max_display=10)
    """

    def __init__(self, handler: "SHAPHandler", show: bool = True):
        self.handler = handler
        self.show = show

    def _maybe_show(self, show: Optional[bool]):
        """Call plt.show() unless explicitly suppressed (Spyder-friendly)."""
        do_show = self.show if show is None else show
        if do_show:
            plt.show()

    def _check_computed(self):
        if self.handler.shap_dict_ is None:
            raise RuntimeError(
                "You must call handler.compute_shap_values() before plotting!"
            )

    def plot_summary_aggregated(
        self,
        max_display: int = 20,
        min_selection_frac: float = 0.0,
        save_path: Optional[str] = None,
        show: Optional[bool] = None,
    ):
        """
        Plot an aggregated SHAP summary across all repetitions and folds,
        for the class selected in `handler.compute_shap_values(target_class=...)`.

        NaNs (features not selected in a given fold) are excluded from the
        importance ranking calculation rather than treated as zero
        contribution, since the two cases are semantically different.
        Only for the plot itself NaNs are filled with 0 (shap.summary_plot
        requires a dense matrix), but the ranking uses nan-aware statistics.

        Parameters
        ----------
        max_display : int, default=20
            Maximum number of features to display in the plot.
        min_selection_frac : float, default=0.0
            Minimum fraction of samples (across all repetitions/folds) for
            which a feature must have a non-NaN SHAP value to be considered
            in the ranking and shown in the plot. E.g. 0.5 keeps only
            features selected in folds covering at least 50% of samples.
            Default 0.0 keeps all features (no filtering), preserving
            previous behaviour.
        save_path : str, optional
            If provided, saves the plot to the specified path.
        show : bool, optional
            Overrides the plotter's default `show` behaviour for this call.

        Returns
        -------
        top_features : pd.DataFrame
            DataFrame with columns "MeanAbsSHAP", "SelectionCount", and
            "SelectionFrac", filtered by `min_selection_frac` and ranked
            by MeanAbsSHAP descending.
        """
        self._check_computed()

        shap_dict = self.handler.shap_dict_
        X = self.handler.X_

        # Concatenate all repetitions
        df_shap_all = pd.concat([shap_dict[r] for r in shap_dict], axis=0)
        n_total_samples = len(df_shap_all)

        # --- Ranking: ignore NaNs, don't treat them as zero contribution ---
        mean_abs_shap = df_shap_all.abs().mean(axis=0, skipna=True)
        selection_count = df_shap_all.notna().sum(axis=0)
        selection_frac = selection_count / n_total_samples

        feature_importance = pd.DataFrame({
            "MeanAbsSHAP": mean_abs_shap,
            "SelectionCount": selection_count,
            "SelectionFrac": selection_frac,
        })

        # Apply threshold before ranking/truncating to max_display
        kept_features = feature_importance[
            feature_importance["SelectionFrac"] >= min_selection_frac
        ].index

        if len(kept_features) == 0:
            raise ValueError(
                f"No feature meets min_selection_frac={min_selection_frac}. "
                "Lower the threshold."
            )

        top_features = feature_importance.loc[kept_features].sort_values(
            by="MeanAbsSHAP", ascending=False
        ).head(max_display)

        # --- Plot: only the surviving (and displayed) features ---
        plotted_features = top_features.index
        df_shap_plot = df_shap_all[plotted_features]
        df_features_plot = X.loc[df_shap_plot.index, plotted_features]

        shap_values_concatenated = df_shap_plot.fillna(0).values
        features_values = df_features_plot.fillna(0).values

        plt.title("SHAP Summary Plot - Global case", fontsize=15, loc='center')
        shap.summary_plot(
            shap_values_concatenated,
            features_values,
            feature_names=df_shap_plot.columns,
            max_display=max_display,
            show=False
        )
        if save_path is not None:
            file_path = Path(save_path) / "shap_summary.png"
            plt.savefig(file_path, bbox_inches='tight', dpi=200)

        self._maybe_show(show)

        return top_features