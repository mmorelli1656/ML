# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 09:57:04 2025

@author: mik16
"""

#%% Libreries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


#%% Feature properties visualization

# ==============================================================
# Feature variance histogram
# ==============================================================
def variance_histogram(X, bins=20, percentile=None, save_path=None):
    """
    Scale features, compute their variance, and plot a histogram of feature variances.

    Optionally, a vertical line can be added to indicate a specific percentile value.
    The plot can be saved to a file if a path is provided.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame containing the features.
    bins : int, default=20
        Number of bins to use in the histogram.
    percentile : float or None, optional
        Percentile (0-100) of variance to highlight with a vertical line.
        If None, no percentile line is drawn.
    save_path : str or None, default=None
        File path to save the figure. If None, the figure is displayed instead.

    Returns
    -------
    variances : pandas.Series
        Series containing the variance of each feature in X after Min-Max scaling.
    """
    # Copy the DataFrame to avoid modifying the original data
    X_scaled = X.copy()

    # Apply Min-Max scaling to normalize feature values to [0, 1]
    scaler = MinMaxScaler()
    X_scaled[:] = scaler.fit_transform(X_scaled)

    # Compute variance for each feature (column)
    variances = X_scaled.var()

    # Plot histogram of feature variances
    plt.figure(figsize=(10, 6))
    plt.hist(variances, bins=bins, color='skyblue', edgecolor='black')

    # If percentile is specified, add a vertical line at that percentile value
    if percentile is not None and 0 <= percentile <= 100:
        perc_value = np.percentile(variances, percentile)
        plt.axvline(
            perc_value,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'{percentile}th percentile = {perc_value:.4f}'
        )
        plt.legend(fontsize=12, loc='upper right')

    # Add title and labels
    plt.title('Distribution of Feature Variances', fontsize=16)
    plt.xlabel('Variance', fontsize=14)
    plt.ylabel('Number of features', fontsize=14)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save figure if save_path is provided; otherwise, display it
    if save_path is not None:
        file_path = Path(save_path) / "variance_hist.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return the variances for further use
    return variances


# ==============================================================
# Feature correlation histogram
# ==============================================================
def correlation_histogram(X, bins='auto', threshold=None, method='pearson', save_path=None):
    """
    Compute pairwise correlations between numerical features and plot a histogram.

    Optionally, a vertical line can be added to indicate a correlation threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame containing numerical features.
    bins : int, sequence, or str, default='auto'
        Number of bins or method for histogram calculation (as in plt.hist).
    threshold : float or None, optional
        Correlation value to highlight with a vertical line. Must be between 0 and 1.
    method : {'pearson', 'spearman', 'kendall'}, default='pearson'
        Method to compute correlations.
    save_path : str or None, default=None
        File path to save the figure. If None, the figure is displayed.

    Returns
    -------
    corr_values : pandas.Series
        Series of correlation values (absolute) for each unique feature pair.
    """
    # Check that all columns are numeric
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
        raise TypeError("All columns must be numeric continuous.")

    # Compute correlation matrix
    corr_matrix = X.corr(method=method).abs()

    # Extract upper triangle (without diagonal) as 1D array
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    corr_values = corr_matrix.values[upper_tri_indices]

    # Convert to pandas Series for convenience
    corr_values = pd.Series(corr_values)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(corr_values, bins=bins, color='lightgreen', edgecolor='black')

    # Highlight threshold if specified
    if threshold is not None:
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1.")
        plt.axvline(
            threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold = {threshold:.2f}'
        )
        plt.legend(fontsize=12, loc='upper right')

    # Add labels and grid
    plt.title('Distribution of Feature Correlations', fontsize=16)
    plt.xlabel('Correlation (absolute value)', fontsize=14)
    plt.ylabel('Number of feature pairs', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save figure or show
    if save_path is not None:
        file_path = Path(save_path) / f"{method}_corr_hist.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return corr_values


# ==============================================================
# Eta**2 histogram
# ==============================================================
def eta_squared_histogram(X, y, bins='auto', threshold=None, save_path=None):
    """
    Compute eta squared (η²) for each numeric feature relative to a categorical target
    and plot a histogram of the η² values.

    Optionally, a vertical line can indicate a threshold for feature selection.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame containing numeric features.
    y : array-like of shape (n_samples,)
        Categorical target.
    bins : int, sequence, or str, default='auto'
        Number of bins or method for histogram calculation (as in plt.hist).
    threshold : float or None, default=None
        Threshold of η² to highlight in the histogram (0 <= threshold <= 1).
    save_path : str or None, default=None
        Path to save the figure. If None, the figure is displayed.

    Returns
    -------
    eta_values : pandas.Series
        Series containing the η² values for each feature.
    """
    # Check all columns are numeric
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
        raise TypeError("All columns must be numeric continuous.")

    # Convert target to numpy array
    y = np.array(y)
    
    # Compute eta squared for each feature
    eta_values = []
    for col in X.columns:
        values = X[col].values
        overall_mean = np.mean(values)
        df = pd.DataFrame({'target': y, 'feature': values})
        grouped = df.groupby('target')['feature']
        group_counts = grouped.count()
        group_means = grouped.mean()
        ss_between = np.sum(group_counts * (group_means - overall_mean) ** 2)
        ss_total = np.sum((values - overall_mean) ** 2)
        eta = ss_between / ss_total if ss_total > 0 else 0
        eta_values.append(eta)

    eta_series = pd.Series(eta_values, index=X.columns)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(eta_series, bins=bins, color='skyblue', edgecolor='black')

    # Highlight threshold if provided
    if threshold is not None:
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1.")
        plt.axvline(
            threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold = {threshold:.2f}'
        )
        plt.legend(fontsize=12, loc='upper right')

    # Add labels, title, and grid
    plt.title('Distribution of Eta Squared (η²) Values', fontsize=16)
    plt.xlabel('Eta Squared (η²)', fontsize=14)
    plt.ylabel('Number of features', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save or show figure
    if save_path is not None:
        file_path = Path(save_path) / "eta_squared_hist.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return eta_series
