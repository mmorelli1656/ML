# -*- coding: utf-8 -*-
"""
Machine learning performance evaluator - works with classification and regression.
Author: mik16, revised with improvements
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score
)
from pathlib import Path


class EvaluationMetrics:
    """
    Class for computing evaluation metrics and visualizations for
    binary classification, multiclass classification, and regression.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        DataFrame containing predictions.
        - First column: true labels
        - Following columns: predicted labels (one column per model/fold).
    df_pred_proba : pandas.DataFrame or None, optional (default=None)
        DataFrame containing predicted probabilities (only needed for ROC/AUC).
        - First column: true labels
        - Following columns: predicted probabilities (one column per model/fold).
    task : {"binaryclass", "multiclass", "regression"}, default="binaryclass"
        Type of machine learning task.
    """

    def __init__(self, df_pred, df_pred_proba=None, task="binaryclass"):
        self.df_pred = df_pred
        self.df_pred_proba = df_pred_proba
        self.task = task
        self.y_true = df_pred.iloc[:, 0].values  # Ground truth labels

    # -------------------------------------------------------------------------
    # METRIC COMPUTATION
    # -------------------------------------------------------------------------

    def _binary_classification_metrics(self):
        """Compute metrics for binary classification."""
        metrics_dict = {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "Specificity": [],
            "F1-score": [],
        }

        for col in self.df_pred.columns[1:]:
            y_pred = self.df_pred[col].values

            metrics_dict["Accuracy"].append(accuracy_score(self.y_true, y_pred))
            metrics_dict["Precision"].append(precision_score(self.y_true, y_pred, zero_division=0))
            metrics_dict["Recall"].append(recall_score(self.y_true, y_pred))
            metrics_dict["F1-score"].append(f1_score(self.y_true, y_pred))

            # Specificity = TN / (TN + FP)
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics_dict["Specificity"].append(specificity)

        # ROC-AUC if probabilities provided
        if self.df_pred_proba is not None:
            metrics_dict["AUC"] = []
            for col in self.df_pred_proba.columns[1:]:
                metrics_dict["AUC"].append(roc_auc_score(self.y_true, self.df_pred_proba[col]))

        return metrics_dict

    def _multiclass_classification_metrics(self):
        """Compute metrics for multiclass classification."""
        metrics_dict = {
            "Accuracy": [],
            "Macro Precision": [],
            "Macro Recall": [],
            "Macro F1-score": [],
        }

        for col in self.df_pred.columns[1:]:
            y_pred = self.df_pred[col].values

            metrics_dict["Accuracy"].append(accuracy_score(self.y_true, y_pred))
            metrics_dict["Macro Precision"].append(
                precision_score(self.y_true, y_pred, average="macro", zero_division=0)
            )
            metrics_dict["Macro Recall"].append(recall_score(self.y_true, y_pred, average="macro"))
            metrics_dict["Macro F1-score"].append(f1_score(self.y_true, y_pred, average="macro"))

        return metrics_dict

    def _regression_metrics(self):
        """Compute metrics for regression."""
        metrics_dict = {"MAE": [], "MAPE": [], "RMSE": [], "R2": []}

        for col in self.df_pred.columns[1:]:
            y_pred = self.df_pred[col].values

            metrics_dict["MAE"].append(mean_absolute_error(self.y_true, y_pred))
            metrics_dict["MAPE"].append(mean_absolute_percentage_error(self.y_true, y_pred))
            metrics_dict["RMSE"].append(np.sqrt(mean_squared_error(self.y_true, y_pred)))
            metrics_dict["R2"].append(r2_score(self.y_true, y_pred))

        return metrics_dict

    def compute_metrics(self):
        """
        Compute metrics based on task type and print summary statistics.

        Returns
        -------
        df_metrics : pandas.DataFrame
            DataFrame containing computed metrics for each prediction column.
        """
        if self.task == "binaryclass":
            metrics_dict = self._binary_classification_metrics()
        elif self.task == "multiclass":
            metrics_dict = self._multiclass_classification_metrics()
        elif self.task == "regression":
            metrics_dict = self._regression_metrics()
        else:
            raise ValueError("Task not recognized! Use 'binaryclass', 'multiclass' or 'regression'.")

        df_metrics = pd.DataFrame(metrics_dict)

        print("\n--- Metrics Results ---")
        for metric, values in metrics_dict.items():
            median_value = np.median(values)
            iqr_value = np.percentile(values, 75) - np.percentile(values, 25)
            mean_value = np.mean(values)
            std_value = np.std(values, ddof=1)  # sample standard deviation

            print(f"{metric}: Median = {median_value:.4f}, IQR = {iqr_value:.4f}")
            print(f"{metric}: Mean = {mean_value:.4f}, Std = {std_value:.4f}")

        return df_metrics

    # -------------------------------------------------------------------------
    # VISUALIZATION METHODS
    # -------------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        perc: str = "row",
        stat_method: str = "median_iqr",
        classes=None,
        save_path: str | Path | None = None,
        palette: str = "YlGnBu",
        annotation_size: int = 12
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
            Class labels for axis ticks.
        save_path : str or Path, optional
            Directory where the figure will be saved.
        palette : str, default="YlGnBu"
            Color palette for heatmap.
        """
        if self.task not in {"binaryclass", "multiclass"}:
            raise ValueError("Confusion matrix is only available for classification tasks.")

        # Compute all confusion matrices once
        y_true = self.df_pred.iloc[:, 0]
        cm_list = [confusion_matrix(y_true, self.df_pred[col]) for col in self.df_pred.columns[1:]]
        cm_array = np.array(cm_list)  # shape: (n_models, n_classes, n_classes)

        if stat_method == "median_iqr":
            stat_cm = np.median(cm_array, axis=0)
            error_cm = np.percentile(cm_array, 75, axis=0) - np.percentile(cm_array, 25, axis=0)
            title_stat = "Median ± IQR"
        elif stat_method == "mean_std":
            stat_cm = cm_array.mean(axis=0)
            error_cm = cm_array.std(axis=0, ddof=1)
            title_stat = "Mean ± Std"
        else:
            raise ValueError("stat_method must be 'median_iqr' or 'mean_std'.")

        # Prepare annotations
        if perc == "row":
            denom = stat_cm.sum(axis=1, keepdims=True)
        elif perc == "total":
            denom = stat_cm.sum()
        else:
            raise ValueError("perc must be 'row' or 'total'.")

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

        # Plot heatmap
        if classes is None:
            classes = np.unique(self.y_true)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            stat_cm,
            annot=annotations,
            fmt="",
            cmap=palette,
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={"size": annotation_size},
        )
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.title(f"Confusion Matrix ({title_stat})", fontsize=16)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "confusion_matrix.png"
            plt.savefig(file_path, bbox_inches="tight")
            print(f"Confusion matrix saved to: {file_path}")

        plt.show()
        plt.close()
    
    def plot_roc_curve(self,
                       stat_method: str = "median_iqr",  # "median_iqr" o "mean_std"
                       save_path=None):
        """
        Plot delle curve ROC per ciascun set di predizioni probabilistiche
        contenute in self.df_pred_proba. Si assume che la prima colonna
        sia 'true_labels' e le successive siano le predizioni.
        
        Parameters
        ----------
        stat_method : {"median_iqr", "mean_std"}, default "median_iqr"
            Metodo per calcolare la statistica AUC e l'errore:
            - "median_iqr": mediana ± interquartile range (IQR)
            - "mean_std": media ± deviazione standard
        save_path : str | Path | None, default None
            Percorso per salvare la figura. Se None la figura non viene salvata.
        """
    
        if not hasattr(self, "df_pred_proba") or self.df_pred_proba is None:
            raise AttributeError("df_pred_proba non è stato trovato o è None.")
    
        if not hasattr(self, "task") or self.task != "binaryclass":
            raise ValueError("Il metodo plot_roc_curve è disponibile solo per problemi di classificazione binaria.")
    
        plt.figure(figsize=(8, 6))
    
        # Plot curva random
        plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
    
        auc_list = []
    
        # ROC curves rosse senza label
        for col in self.df_pred_proba.columns[1:]:
            y_scores = self.df_pred_proba[col]
            fpr, tpr, _ = roc_curve(self.y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='red', alpha=0.4)
            auc_list.append(roc_auc)
    
        auc_list = np.array(auc_list)
    
        if stat_method == "median_iqr":
            auc_stat = np.median(auc_list)
            auc_err = np.percentile(auc_list, 75) - np.percentile(auc_list, 25)
            label = f"Median AUC = {auc_stat:.2f} ± {auc_err:.2f} (IQR)"
        elif stat_method == "mean_std":
            auc_stat = auc_list.mean()
            auc_err = auc_list.std(ddof=1)
            label = f"Mean AUC = {auc_stat:.2f} ± {auc_err:.2f} (Std)"
        else:
            raise ValueError("stat_method deve essere 'median_iqr' o 'mean_std'.")
    
        # Linee fittizie per la legenda
        plt.plot([], [], color='red', label='ROC')
        plt.plot([], [], color='white', label=label)
    
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
    
        # Salvataggio
        if save_path is not None:
            from pathlib import Path
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "roc.png"
            plt.savefig(file_path)
            print(f"Figura salvata in: {file_path}")
    
        plt.show()
        plt.close()

    def plot_metrics_boxplot(
        self, df_metrics: pd.DataFrame, save_path: str | Path | None = None, palette: str = "Set2"
    ):
        """
        Plot boxplots of evaluation metrics.

        Parameters
        ----------
        df_metrics : pandas.DataFrame
            DataFrame containing computed metrics.
        save_path : str or Path, optional
            Directory where the figure will be saved.
        palette : str, default="Set2"
            Color palette for boxplots.
        """
        if df_metrics is None or df_metrics.empty:
            raise ValueError("df_metrics cannot be empty.")

        if self.task == "regression":
            metrics = ["MAE", "MAPE", "RMSE", "R2"]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, metric in enumerate(metrics):
                sns.boxplot(
                    y=df_metrics[metric],
                    ax=axes[i],
                    color=sns.color_palette(palette)[i],
                    showfliers=True,
                    width=0.3,
                )
                axes[i].set_title(metric, fontsize=16)
                axes[i].set_ylabel("Values", fontsize=12)

            plt.suptitle("Regression Metrics Boxplots", fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 1])

        else:
            df_melted = df_metrics.melt(var_name="Metric", value_name="Value")
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x="Metric",
                y="Value",
                data=df_melted,
                hue="Metric",
                palette=palette,
                legend=False,
            )
            plt.xlabel("Metrics", fontsize=14)
            plt.ylabel("Values", fontsize=14)
            plt.title("Classification Metrics Boxplot", fontsize=16)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "metrics_boxplot.png"
            plt.savefig(file_path, bbox_inches="tight")
            print(f"Boxplot saved to: {file_path}")

        plt.show()
        plt.close()
