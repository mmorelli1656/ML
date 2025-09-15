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
        plt.legend(fontsize=12)

    # Add title and labels
    plt.title('Distribution of feature variances', fontsize=16)
    plt.xlabel('Variance', fontsize=14)
    plt.ylabel('Number of features', fontsize=14)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save figure if save_path is provided; otherwise, display it
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return the variances for further use
    return variances
