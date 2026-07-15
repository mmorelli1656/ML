# -*- coding: utf-8 -*-
"""
Machine learning performance evaluator - works with classification and regression.

This module is split into two classes with a single responsibility each:

- ``MetricsEvaluator``: computes metrics only (no plotting side effects).
- ``MetricsPlotter``: takes a ``MetricsEvaluator`` instance and produces
  visualizations. Every plotting method returns the ``(fig, ax)`` objects it
  created and never calls ``plt.show()`` internally, so figures can be
  composed into external grids, saved, or customized further by the caller.

Author: mik16
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score
)
from pathlib import Path

VALID_TASKS = {"binaryclass", "multiclass", "regression"}

# -----------------------------------------------------------------------
# Default metric registries: {task: {metric_name: fn(y_true, y_pred) -> float}}
# These are the metrics computed automatically for each task. Extend them
# per-instance via `extra_metrics`, or silence some via `exclude_metrics`
# (see MetricsEvaluator.__init__).
# -----------------------------------------------------------------------
_DEFAULT_METRICS = {
    "binaryclass": {
        "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "Recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "F1-score": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        # Specificity is added per-instance in _build_metric_registry() since
        # it needs access to self.classes for a stable confusion_matrix shape.
    },
    "multiclass": {
        "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "Macro Precision": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
        "Macro Recall": lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
        "Macro F1-score": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    },
    "regression": {
        "MAE": lambda yt, yp: mean_absolute_error(yt, yp),
        "MAPE": lambda yt, yp: mean_absolute_percentage_error(yt, yp),
        "RMSE": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        "R2": lambda yt, yp: r2_score(yt, yp),
    },
}

# Metric registries that need predicted *probabilities* instead of hard
# predictions: {task: {metric_name: fn(y_true, y_proba) -> float}}
_DEFAULT_PROBA_METRICS = {
    "binaryclass": {
        "AUC": lambda yt, yproba: roc_auc_score(yt, yproba),
    },
}


# =============================================================================
# METRICS
# =============================================================================

class MetricsEvaluator:
    """
    Class for computing evaluation metrics for binary classification,
    multiclass classification, and regression.

    Configuration (``task``) is set at construction time; data is provided
    when calling :meth:`compute_metrics`, which both stores it on the
    instance (so :meth:`compute_confusion_matrices` and
    :class:`MetricsPlotter` can reuse it) and returns the computed metrics
    immediately - no separate "fit" step required.

    The set of computed metrics is a registry, not a hardcoded list: use
    ``extra_metrics``/``extra_proba_metrics`` to add custom metrics, and
    ``exclude_metrics`` to drop any metric (built-in or custom) by name.

    Parameters
    ----------
    task : {"binaryclass", "multiclass", "regression"}, default="binaryclass"
        Type of machine learning task.
    extra_metrics : dict of {str: callable}, optional
        Additional metrics to compute from hard predictions, as
        ``{name: fn(y_true, y_pred) -> float}``. If a name matches a
        built-in metric, this overrides it.
    extra_proba_metrics : dict of {str: callable}, optional
        Additional metrics to compute from predicted probabilities
        (requires ``df_pred_proba`` to be provided), as
        ``{name: fn(y_true, y_proba) -> float}``.
    exclude_metrics : list of str, optional
        Names of metrics (built-in or custom) to leave out of the results.

    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> ev = MetricsEvaluator(
    ...     task="binaryclass",
    ...     extra_metrics={"Balanced Accuracy": balanced_accuracy_score},
    ...     exclude_metrics=["Specificity"],
    ... )
    >>> df_metrics = ev.compute_metrics(y_true, df_pred, df_pred_proba)
    """

    def __init__(
        self,
        task="binaryclass",
        extra_metrics=None,
        extra_proba_metrics=None,
        exclude_metrics=None,
    ):
        # Fail fast on invalid task instead of only failing later inside a
        # compute/plot call.
        if task not in VALID_TASKS:
            raise ValueError(
                f"Task not recognized: '{task}'. Use one of {sorted(VALID_TASKS)}."
            )
        self.task = task
        self.extra_metrics = dict(extra_metrics or {})
        self.extra_proba_metrics = dict(extra_proba_metrics or {})
        self.exclude_metrics = set(exclude_metrics or [])
        self._is_fitted = False

    def _build_metric_registry(self):
        """Assemble the {name: fn(y_true, y_pred)} registry for this task."""
        registry = dict(_DEFAULT_METRICS.get(self.task, {}))

        if self.task == "binaryclass":
            # Needs self.classes for a stable confusion_matrix shape, so it's
            # bound here (after data is loaded) rather than in the module-level
            # default registry.
            registry["Specificity"] = self._specificity

        registry.update(self.extra_metrics)

        for name in self.exclude_metrics:
            registry.pop(name, None)

        return registry

    def _build_proba_metric_registry(self):
        """Assemble the {name: fn(y_true, y_proba)} registry for this task."""
        registry = dict(_DEFAULT_PROBA_METRICS.get(self.task, {}))
        registry.update(self.extra_proba_metrics)

        for name in self.exclude_metrics:
            registry.pop(name, None)

        return registry

    def _specificity(self, y_true, y_pred):
        """Specificity = TN / (TN + FP), using a fixed label order for a
        consistent confusion_matrix shape across folds/models."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=self.classes).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def _load_data(self, y_true, df_pred, df_pred_proba=None):
        """
        Validate and store the data needed to compute metrics/plots.
        Called internally by :meth:`compute_metrics`, not meant to be called
        directly by users.
        """
        y_true = np.asarray(y_true)

        if len(y_true) != df_pred.shape[0]:
            raise ValueError(
                f"y_true has {len(y_true)} samples but df_pred has {df_pred.shape[0]} rows."
            )

        if df_pred_proba is not None:
            if len(y_true) != df_pred_proba.shape[0]:
                raise ValueError(
                    f"y_true has {len(y_true)} samples but df_pred_proba has "
                    f"{df_pred_proba.shape[0]} rows."
                )
            if df_pred.shape[1] != df_pred_proba.shape[1]:
                raise ValueError(
                    "df_pred and df_pred_proba must have the same number of "
                    f"model/fold columns (got {df_pred.shape[1]} and {df_pred_proba.shape[1]})."
                )

        self.y_true = y_true
        self.df_pred = df_pred
        self.df_pred_proba = df_pred_proba
        self.classes = np.unique(y_true)
        self._is_fitted = True

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "This MetricsEvaluator instance has no data yet. "
                "Call compute_metrics(y_true, df_pred, df_pred_proba=None) first."
            )

    # -------------------------------------------------------------------------
    # METRIC COMPUTATION
    # -------------------------------------------------------------------------

    def compute_metrics(self, y_true, df_pred, df_pred_proba=None, verbose: bool = True):
        """
        Load data and compute metrics based on task type, in a single call.
        The data is also kept on the instance so that
        :meth:`compute_confusion_matrices` and :class:`MetricsPlotter` can
        reuse it without it being passed again.

        Which metrics get computed is driven by the registry assembled from
        the task's defaults plus ``extra_metrics``/``extra_proba_metrics``,
        minus anything in ``exclude_metrics`` (all set in ``__init__``).

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground-truth labels/values.
        df_pred : pandas.DataFrame of shape (n_samples, n_models)
            One column per model/fold; every column is a set of predictions.
        df_pred_proba : pandas.DataFrame of shape (n_samples, n_models), optional
            One column per model/fold of predicted probabilities (only
            needed for ROC/AUC-style metrics, binary classification only).
        verbose : bool, default=True
            If True, print median/IQR and mean/std for each metric.

        Returns
        -------
        df_metrics : pandas.DataFrame
            DataFrame containing computed metrics for each prediction column.
        """
        self._load_data(y_true, df_pred, df_pred_proba)

        metric_registry = self._build_metric_registry()
        metrics_dict = {name: [] for name in metric_registry}

        for col in self.df_pred.columns:
            y_pred = self.df_pred[col].values
            for name, fn in metric_registry.items():
                metrics_dict[name].append(fn(self.y_true, y_pred))

        if self.df_pred_proba is not None:
            proba_registry = self._build_proba_metric_registry()
            for name in proba_registry:
                metrics_dict[name] = []
            for col in self.df_pred_proba.columns:
                y_scores = self.df_pred_proba[col].values
                for name, fn in proba_registry.items():
                    metrics_dict[name].append(fn(self.y_true, y_scores))

        df_metrics = pd.DataFrame(metrics_dict)
        self.df_metrics = df_metrics  # stored so MetricsPlotter can reuse it by default

        if verbose:
            print("\n--- Metrics Results ---")
            for metric, values in metrics_dict.items():
                median_value = np.median(values)
                iqr_value = np.percentile(values, 75) - np.percentile(values, 25)
                mean_value = np.mean(values)
                std_value = np.std(values, ddof=1) if len(values) > 1 else 0.0

                print(f"{metric}: Median = {median_value:.4f}, IQR = {iqr_value:.4f}")
                print(f"{metric}: Mean = {mean_value:.4f}, Std = {std_value:.4f}")

        return df_metrics

    def compute_confusion_matrices(self):
        """
        Compute one confusion matrix per prediction column, all using a
        consistent set of labels (``self.classes``) so that every matrix has
        the same shape regardless of whether a given fold/model predicted
        every class.

        Returns
        -------
        cm_array : numpy.ndarray
            Array of shape (n_models, n_classes, n_classes).
        """
        self._check_fitted()

        if self.task not in {"binaryclass", "multiclass"}:
            raise ValueError("Confusion matrices are only available for classification tasks.")

        cm_list = [
            confusion_matrix(self.y_true, self.df_pred[col], labels=self.classes)
            for col in self.df_pred.columns
        ]
        return np.array(cm_list)


# =============================================================================
# PLOTTING
# =============================================================================

class MetricsPlotter:
    """
    Visualization companion for :class:`MetricsEvaluator`.

    Every method returns the ``(fig, ax)`` (or ``(fig, axes)``) it built.
    By default ``show=True`` so the figure is displayed immediately (needed
    outside Jupyter, e.g. in Spyder or plain scripts, where nothing renders
    without an explicit ``plt.show()``). Pass ``show=False`` if you want to
    embed the figure into a larger layout or customize it further before
    displaying it yourself. Pass ``save_path`` to also persist the figure to
    disk.

    Parameters
    ----------
    evaluator : MetricsEvaluator
        An evaluator instance on which ``compute_metrics(y_true, df_pred, ...)``
        has already been called.
    """

    def __init__(self, evaluator: MetricsEvaluator):
        evaluator._check_fitted()
        self.ev = evaluator

    @staticmethod
    def _maybe_save(fig, save_path, filename):
        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / filename
            fig.savefig(file_path, bbox_inches="tight")
            print(f"Figure saved to: {file_path}")

    @staticmethod
    def _maybe_show(fig, show):
        if show:
            plt.show()

    # -------------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        perc: str = "row",
        stat_method: str = "median_iqr",
        classes=None,
        cm_array: np.ndarray | None = None,
        save_path: str | Path | None = None,
        palette: str = "YlGnBu",
        annotation_size: int = 12,
        show: bool = True,
    ):
        """
        Plot aggregated confusion matrix with uncertainty.

        Parameters
        ----------
        perc : {"row", "total"}, default="row"
            Percentage calculation method.
        stat_method : {"median_iqr", "mean_std"}, default="median_iqr"
            Statistic method to summarize multiple confusion matrices.
        classes : list of str, optional
            Class labels for axis ticks. Defaults to the evaluator's classes.
        cm_array : numpy.ndarray, optional
            Array of shape (n_models, n_classes, n_classes) to plot directly.
            If not provided, falls back to
            ``evaluator.compute_confusion_matrices()``. Pass this explicitly
            only if you want to plot matrices other than the evaluator's own.
        save_path : str or Path, optional
            Directory where the figure will be saved.
        palette : str, default="YlGnBu"
            Color palette for heatmap.
        annotation_size : int, default=12
            Font size for cell annotations.
        show : bool, default=True
            If True, call ``plt.show()`` before returning (needed to see the
            plot outside Jupyter, e.g. in Spyder or a plain script). Set to
            False if you plan to display/compose the figure yourself.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if cm_array is None:
            cm_array = self.ev.compute_confusion_matrices()

        if stat_method == "median_iqr":
            stat_cm = np.median(cm_array, axis=0)
            error_cm = np.percentile(cm_array, 75, axis=0) - np.percentile(cm_array, 25, axis=0)
            title_stat = "Median ± IQR"
        elif stat_method == "mean_std":
            stat_cm = cm_array.mean(axis=0)
            error_cm = cm_array.std(axis=0, ddof=1) if cm_array.shape[0] > 1 else np.zeros_like(stat_cm)
            title_stat = "Mean ± Std"
        else:
            raise ValueError("stat_method must be 'median_iqr' or 'mean_std'.")

        if perc == "row":
            denom = stat_cm.sum(axis=1, keepdims=True)
        elif perc == "total":
            denom = stat_cm.sum()
        else:
            raise ValueError("perc must be 'row' or 'total'.")

        denom = np.where(denom == 0, 1, denom)  # avoid divide-by-zero
        stat_perc = stat_cm / denom * 100
        error_perc = error_cm / denom * 100

        annotations = [
            [
                f"{stat_cm[i, j]:.2f} ± {error_cm[i, j]:.2f}\n"
                f"{stat_perc[i, j]:.2f} ± {error_perc[i, j]:.2f} %"
                for j in range(stat_cm.shape[1])
            ]
            for i in range(stat_cm.shape[0])
        ]

        if classes is None:
            classes = self.ev.classes

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            stat_cm,
            annot=annotations,
            fmt="",
            cmap=palette,
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={"size": annotation_size},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("Actual", fontsize=14)
        ax.set_title(f"Confusion Matrix ({title_stat})", fontsize=16)
        fig.tight_layout()

        self._maybe_save(fig, save_path, "confusion_matrix.png")
        self._maybe_show(fig, show)
        return fig, ax

    def plot_roc_curve(
        self,
        stat_method: str = "median_iqr",
        y_true=None,
        df_pred_proba: pd.DataFrame | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ):
        """
        Plot ROC curves for every set of probabilistic predictions contained
        in ``df_pred_proba`` (one column per model/fold), against the
        ground-truth labels.

        Parameters
        ----------
        stat_method : {"median_iqr", "mean_std"}, default="median_iqr"
            Method used to summarize the AUC values and their spread:
            - "median_iqr": median ± interquartile range (IQR)
            - "mean_std": mean ± standard deviation
        y_true : array-like, optional
            Ground-truth labels to plot against. If not provided, falls back
            to the evaluator's own ``y_true``.
        df_pred_proba : pandas.DataFrame, optional
            Probabilities to plot, one column per model/fold. If not
            provided, falls back to the evaluator's own ``df_pred_proba``.
            Pass these two explicitly only if you want to plot data other
            than the evaluator's own.
        save_path : str or Path, optional
            Directory where the figure will be saved.
        show : bool, default=True
            If True, call ``plt.show()`` before returning.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        ev = self.ev

        if y_true is None:
            y_true = ev.y_true
        if df_pred_proba is None:
            df_pred_proba = ev.df_pred_proba

        if df_pred_proba is None:
            raise AttributeError("df_pred_proba was not provided (it is None).")

        if ev.task != "binaryclass":
            raise ValueError("plot_roc_curve is only available for binary classification.")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Random-chance reference line
        ax.plot([0, 1], [0, 1], "k--", label="Random Model")

        auc_list = []
        for col in df_pred_proba.columns:
            y_scores = df_pred_proba[col]
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="red", alpha=0.4)
            auc_list.append(roc_auc)

        auc_list = np.array(auc_list)

        if stat_method == "median_iqr":
            auc_stat = np.median(auc_list)
            auc_err = np.percentile(auc_list, 75) - np.percentile(auc_list, 25)
            stat_label = f"Median AUC = {auc_stat:.2f} ± {auc_err:.2f} (IQR)"
        elif stat_method == "mean_std":
            auc_stat = auc_list.mean()
            auc_err = auc_list.std(ddof=1) if len(auc_list) > 1 else 0.0
            stat_label = f"Mean AUC = {auc_stat:.2f} ± {auc_err:.2f} (Std)"
        else:
            raise ValueError("stat_method must be 'median_iqr' or 'mean_std'.")

        # Build the legend explicitly instead of relying on an invisible
        # "white" line (which only disappears on a white background).
        legend_handles = [
            Line2D([0], [0], color="black", linestyle="--", label="Random Model"),
            Line2D([0], [0], color="red", alpha=0.4, label="ROC"),
            Line2D([0], [0], color="none", label=stat_label),
        ]
        ax.legend(handles=legend_handles, fontsize=12)

        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title("ROC Curve", fontsize=16)
        ax.grid(True)
        fig.tight_layout()

        self._maybe_save(fig, save_path, "roc.png")
        self._maybe_show(fig, show)
        return fig, ax

    def plot_class_probabilities(
        self,
        threshold: float = 0.5,
        bins: int | str = 30,
        labels: list[str] | None = None,
        palette: list[str] = ["skyblue", "salmon"],
        y_true=None,
        df_pred_proba: pd.DataFrame | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ):
        """
        Plot probability distributions of the two classes for a binary
        classification problem, supporting multiple repetitions of
        cross-validation by flattening all probability columns.

        Parameters
        ----------
        threshold : float, default=0.5
            Threshold to separate the predicted classes.
        bins : int or {"auto", "fd", "sturges", ...}, default=30
            Number of bins, or any strategy name accepted by
            ``numpy.histogram_bin_edges`` (e.g. "auto"). Passing an int ``n``
            produces exactly ``n`` bins (n+1 edges) - previously this
            silently produced ``n - 1`` bins via ``np.linspace``.
        labels : list of str, optional
            Names of the two classes. If None, defaults to ['Class 0', 'Class 1'].
        palette : list of str, default=["skyblue", "salmon"]
            Colors for the histograms of the two classes.
        y_true : array-like, optional
            Ground-truth labels to plot against. If not provided, falls back
            to the evaluator's own ``y_true``.
        df_pred_proba : pandas.DataFrame, optional
            Probabilities to plot, one column per model/fold. If not
            provided, falls back to the evaluator's own ``df_pred_proba``.
            Pass these two explicitly only if you want to plot data other
            than the evaluator's own.
        save_path : str or Path, optional
            Directory to save the figure.
        show : bool, default=True
            If True, call ``plt.show()`` before returning.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        ev = self.ev

        if ev.task != "binaryclass":
            raise ValueError("plot_class_probabilities is only available for binary classification.")

        if y_true is None:
            y_true = ev.y_true
        if df_pred_proba is None:
            df_pred_proba = ev.df_pred_proba

        if df_pred_proba is None:
            raise ValueError("df_pred_proba must be provided for plotting probabilities.")

        classes = np.unique(y_true)

        if labels is None:
            labels = ["Class 0", "Class 1"]

        # Flatten all probability columns (all CV repetitions)
        y_scores = df_pred_proba.values.flatten()

        # Repeat the true labels for each repetition
        n_reps = df_pred_proba.shape[1]
        y_true_repeated = np.repeat(y_true, n_reps)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Compute common bin edges for both classes.
        # np.histogram_bin_edges handles both integer bin counts (producing
        # exactly `bins` bins) and named strategies like "auto"/"fd"/"sturges".
        bin_edges = np.histogram_bin_edges(y_scores, bins=bins)

        for cls_idx, cls_value in enumerate(classes):
            cls_scores = y_scores[y_true_repeated == cls_value]
            ax.hist(
                cls_scores,
                bins=bin_edges,
                color=palette[cls_idx % len(palette)],
                alpha=0.6,
                label=f"{labels[cls_idx]}",
                edgecolor="none",
            )

        ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Threshold = {threshold}")

        ax.set_xlabel("Predicted Probability", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title("Predicted Probability Distribution per Class", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        self._maybe_save(fig, save_path, "class_probabilities.png")
        self._maybe_show(fig, show)
        return fig, ax

    def plot_metrics_boxplot(
        self,
        df_metrics: pd.DataFrame | None = None,
        save_path: str | Path | None = None,
        palette: str = "Set2",
        show: bool = True,
    ):
        """
        Plot boxplots of evaluation metrics.

        Parameters
        ----------
        df_metrics : pandas.DataFrame, optional
            DataFrame containing computed metrics. If not provided, falls
            back to the metrics last computed by the evaluator's
            ``compute_metrics()`` call (``evaluator.df_metrics``). Pass this
            explicitly only if you want to plot a different set of metrics.
        save_path : str or Path, optional
            Directory where the figure will be saved.
        palette : str, default="Set2"
            Color palette for boxplots.
        show : bool, default=True
            If True, call ``plt.show()`` before returning.

        Returns
        -------
        fig, axes : matplotlib Figure and Axes (single Axes for
            classification tasks, array of Axes for regression - one per
            metric column in df_metrics).
        """
        if df_metrics is None:
            if not hasattr(self.ev, "df_metrics"):
                raise ValueError(
                    "No df_metrics available. Call compute_metrics() on the "
                    "evaluator first, or pass df_metrics explicitly."
                )
            df_metrics = self.ev.df_metrics

        if df_metrics is None or df_metrics.empty:
            raise ValueError("df_metrics cannot be empty.")

        if self.ev.task == "regression":
            # Metric columns are read from df_metrics itself (not hardcoded),
            # so this stays correct even if extra/excluded regression metrics
            # change which columns are present.
            metrics = list(df_metrics.columns)
            n_metrics = len(metrics)
            n_cols = 2
            n_rows = int(np.ceil(n_metrics / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            colors = sns.color_palette(palette)
            for i, metric in enumerate(metrics):
                sns.boxplot(
                    y=df_metrics[metric],
                    ax=axes[i],
                    color=colors[i % len(colors)],
                    showfliers=True,
                    width=0.3,
                )
                axes[i].set_title(metric, fontsize=16)
                axes[i].set_ylabel("Values", fontsize=12)

            # Hide any unused subplot cells (e.g. 3 metrics in a 2x2 grid)
            for j in range(n_metrics, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle("Metrics Boxplots", fontsize=20)
            fig.tight_layout(rect=[0, 0.03, 1, 1])

        else:
            df_melted = df_metrics.melt(var_name="Metric", value_name="Value")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x="Metric",
                y="Value",
                data=df_melted,
                hue="Metric",
                palette=palette,
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Metrics", fontsize=14)
            ax.set_ylabel("Values", fontsize=14)
            ax.set_title("Metrics Boxplot", fontsize=16)
            ax.tick_params(axis="x", rotation=45, labelsize=12)
            fig.tight_layout()
            axes = ax

        self._maybe_save(fig, save_path, "metrics_boxplot.png")
        self._maybe_show(fig, show)
        return fig, axes