# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 12:22:39 2025

@author: mik16
"""

#%% Libreries and modules

import sys
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from pathlib import Path

# Import modules
sys.path.append(str(Path.home() / "Github" / "ML"))
sys.path.append(str(Path.home() / "Github" / "Utils"))

from performance_evaluator import EvaluationMetrics
from elapsed_timer import Timer


#%% Functions

def generate_example_datasets(
    n_samples: int = 10,
    n_preds: int = 5,
    tasks: Union[str, List[str]] = "all",
    random_state: int = 42
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generate example datasets for binary classification, binary classification with probabilities,
    multiclass classification, and regression. The number of prediction columns and samples is dynamic.

    Parameters
    ----------
    n_samples : int, default=10
        Number of rows (samples) in each dataset.
    n_preds : int, default=5
        Number of prediction columns to generate.
    tasks : {'binary', 'binary_proba', 'multiclass', 'regression', 'all'} or list, default='all'
        Which datasets to generate. Can be:
        - 'binary' : Binary classification dataset
        - 'binary_proba' : Binary classification with probabilities
        - 'multiclass' : Multiclass classification dataset
        - 'regression' : Regression dataset
        - 'all' : Generate all datasets
        - list of the above
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    result : pandas.DataFrame or dict of {str: pandas.DataFrame}
        - If a single dataset is requested, returns that DataFrame.
        - If multiple datasets are requested, returns a dictionary.
    """
    np.random.seed(random_state)

    # Normalize tasks input
    if isinstance(tasks, str):
        if tasks == "all":
            tasks = ["binary", "binary_proba", "multiclass", "regression"]
        else:
            tasks = [tasks]

    datasets = {}

    # ---------------- Binary classification ----------------
    if "binary" in tasks or "binary_proba" in tasks:
        true_labels_binary = np.random.choice([0, 1], size=n_samples)

    if "binary" in tasks:
        preds = pd.DataFrame(
            np.random.choice([0, 1], size=(n_samples, n_preds)),
            columns=[f"pred{i}" for i in range(1, n_preds + 1)]
        )
        df_binary = pd.concat([pd.DataFrame({'Labels': true_labels_binary}), preds], axis=1)
        datasets["binary"] = df_binary

    if "binary_proba" in tasks:
        pred_array = np.zeros((n_samples, n_preds))
        for i in range(n_preds):
            mean_val = 0.3 + (i + 1) * 0.2
            pred_array[:, i] = np.clip(
                np.random.normal(loc=mean_val, scale=0.15, size=n_samples),
                0, 1
            )
        preds_proba = pd.DataFrame(pred_array, columns=[f"pred{i}" for i in range(1, n_preds + 1)])
        df_binary_proba = pd.concat([pd.DataFrame({'Labels': true_labels_binary}), preds_proba], axis=1)
        datasets["binary_proba"] = df_binary_proba

    # ---------------- Multiclass classification ----------------
    if "multiclass" in tasks:
        true_labels_multi = np.random.choice([0, 1, 2], size=n_samples)
        preds = pd.DataFrame(
            np.random.choice([0, 1, 2], size=(n_samples, n_preds)),
            columns=[f"pred{i}" for i in range(1, n_preds + 1)]
        )
        df_multi = pd.concat([pd.DataFrame({'Labels': true_labels_multi}), preds], axis=1)
        datasets["multiclass"] = df_multi

    # ---------------- Regression ----------------
    if "regression" in tasks:
        true_labels_reg = np.random.uniform(0, 100, n_samples)
        pred_array = np.zeros((n_samples, n_preds))
        for i in range(n_preds):
            noise = np.random.normal(0, 2 + (i + 1) * 2, n_samples)
            pred_array[:, i] = true_labels_reg + noise
        preds_reg = pd.DataFrame(pred_array, columns=[f"pred{i}" for i in range(1, n_preds + 1)])
        df_reg = pd.concat([pd.DataFrame({'Labels': true_labels_reg}), preds_reg], axis=1)
        datasets["regression"] = df_reg

    # Return single DataFrame if only one task was requested
    if len(datasets) == 1:
        return next(iter(datasets.values()))
    return datasets


#%% Datasets parameters

# Number of samples
n_samples = int(input("Enter the number of rows (samples) for each dataset: "))
# Number of repeated CV n_repeats
n_preds = int(input("Enter the number of prediction columns: "))


#%% Binary classification (with probabilities)

# Binary classification dataset
df_binaryclass = generate_example_datasets(
    n_samples=n_samples,
    n_preds=n_preds,
    tasks="binary"
)

# Binary classification with predicted probabilities
df_binaryclass_proba = generate_example_datasets(
    n_samples=n_samples,
    n_preds=n_preds,
    tasks="binary_proba"
)

# Initialize evaluator
evaluator = EvaluationMetrics(df_pred=df_binaryclass, df_pred_proba=df_binaryclass_proba,
                              task="binaryclass")

# Compute perfomances metrics
df_metrics = evaluator.compute_metrics()

# Plot confusion matrix with labels
classes_name = ['Test1', 'Test2']
with Timer():
    evaluator.plot_confusion_matrix(
        perc='row',
        stat_method="mean_std",
        classes=classes_name,
        save_path=None)

# Plot ROC curves
evaluator.plot_roc_curve(save_path=None)

# Plot metrics boxplots
evaluator.plot_metrics_boxplot(df_metrics, save_path=None)

del classes_name, evaluator


#%% Evaluator classificazione multiclasse

# Multiclass classification dataset
df_multiclass = generate_example_datasets(
    n_samples=n_samples,
    n_preds=n_preds,
    tasks="multiclass"
)

# Initialize evaluator
evaluator = EvaluationMetrics(df_multiclass, df_pred_proba=None,
                              task="multiclass")

# Compute perfomances metrics
df_metrics = evaluator.compute_metrics()

# Plot confusion matrix with labels
classes_name = ['Test1', 'Test2', 'Test3']
with Timer():
    evaluator.plot_confusion_matrix(perc='row',
                                    stat_method="mean_std",
                                    classes=classes_name,
                                    save_path=None)

del classes_name, evaluator


#%% Evaluator regressione

# Regression dataset
df_regression = generate_example_datasets(
    n_samples=n_samples,
    n_preds=n_preds,
    tasks="regression"
)

# Initialize evaluator
evaluator = EvaluationMetrics(df_regression, df_pred_proba=None, task="regression")

# Compute perfomances metrics
df_metrics = evaluator.compute_metrics()

# Plot metrics boxplots
evaluator.plot_metrics_boxplot(df_metrics, save_path=None)

del evaluator