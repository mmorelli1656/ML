# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 09:22:42 2025

@author: mik16
"""

#%% Libraries and submodules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter, find_peaks
from sklearn.mixture import GaussianMixture
    
# Submodules utilities
from Utils.project_paths import ProjectPaths


#%% Functions

# ==============================================================
# 1) Preprocessing of spectra: interpolation, smoothing, normalization, peak detection
# ==============================================================
def preprocess_spectra(spectra_dict, min_wn=200, max_wn=2000, step=1,
                       sg_window=91, sg_order=3, peak_threshold=0.0001,
                       show_plots=False):
    """
    Preprocess all spectra by:
      - Interpolating intensity values onto a common wavenumber grid using monotone spline (PCHIP).
      - Applying Savitzky-Golay smoothing.
      - Normalizing each spectrum so that total area under curve = 1.
      - Detecting peaks above a given prominence threshold.

    Parameters
    ----------
    spectra_dict : dict
        Dictionary with {patient: {"Wave": array, "Intensity": array}}
    min_wn, max_wn, step : int
        Common grid definition (wavenumber range and spacing).
    sg_window, sg_order : int
        Savitzky-Golay filter parameters.
    peak_threshold : float
        Minimum prominence for peak detection.
    show_plot : bool
        If True, show diagnostic plots for each spectrum.

    Returns
    -------
    all_normalized_spectra : DataFrame
        Each row = normalized spectrum for one patient.
    all_peaks : dict
        {patient: {"Positions": array, "Heights": array}} with detected peaks.
    common_grid : ndarray
        The common wavenumber grid.
    """

    common_grid = np.arange(min_wn, max_wn + 1, step)
    all_normalized_spectra = pd.DataFrame()
    all_peaks = {}

    for patient, data in spectra_dict.items():
        wn = data['Wave'].values
        intensity = data['Intensity'].values

        # Use mask to restrict interpolation only to the range of original data
        mask = (common_grid >= wn.min()) & (common_grid <= wn.max())
        grid_interp = common_grid[mask]

        # PCHIP ensures monotone and shape-preserving interpolation (avoids oscillations)
        pchip = PchipInterpolator(wn, intensity)
        intensity_interp = pchip(grid_interp)

        # Savitzky-Golay smoothing: polynomial filter to reduce noise
        smoothed = savgol_filter(intensity_interp, window_length=sg_window, polyorder=sg_order)

        # Force negative values (from noise or filter) to zero
        smoothed[smoothed < 0] = 0

        # Normalize so total area under the curve = 1
        intensity_norm = smoothed / smoothed.sum()

        # Fill into full grid (outside range → zeros)
        full_spectrum = np.zeros_like(common_grid, dtype=float)
        full_spectrum[mask] = intensity_norm

        # Store normalized spectrum
        full_spectrum_df = pd.Series(full_spectrum).to_frame().T
        full_spectrum_df.index = [patient]
        all_normalized_spectra = pd.concat([all_normalized_spectra, full_spectrum_df])

        # Detect peaks by prominence threshold
        peaks, _ = find_peaks(full_spectrum, prominence=peak_threshold)
        peaks_positions = common_grid[peaks]
        peaks_heights = full_spectrum[peaks]

        all_peaks[patient] = {"Positions": peaks_positions, "Heights": peaks_heights}

        if show_plots:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            axs[0, 0].plot(wn, intensity, color='blue')
            axs[0, 0].set_title('Original')

            axs[0, 1].plot(grid_interp, intensity_interp, color='red')
            axs[0, 1].set_title('Interpolated')

            axs[1, 0].plot(grid_interp, smoothed, color='green')
            axs[1, 0].set_title('Smoothed')

            axs[1, 1].plot(common_grid, full_spectrum, color='brown')
            axs[1, 1].plot(peaks_positions, peaks_heights, 'ro')
            axs[1, 1].set_title('Normalized + Peaks')

            for ax in axs.flat:
                ax.set_xlabel("Wavenumber (cm$^{-1}$)")

            plt.suptitle(patient)
            plt.tight_layout()
            plt.show()

    return all_normalized_spectra, all_peaks, common_grid


# ==============================================================
# 2) Fit Gaussian Mixture Model to all peak positions
# ==============================================================
def fit_gmm(all_peaks, max_components=19, show_plots=False):
    """
    Fit univariate Gaussian Mixture Models (GMM) on concatenated peak positions
    to identify common peaks across spectra.

    - Uses BIC (Bayesian Information Criterion) to select the optimal number of components.
    - Returns Gaussian means, standard deviations, weights, and intervals [mu-sigma, mu+sigma].

    Parameters
    ----------
    all_peaks : dict
        {patient: {"Positions": array}}
    max_components : int
        Maximum number of mixture components to test.
    show_plots : bool
        If True, plot GMM fits for each number of components.

    Returns
    -------
    common_peaks : DataFrame
        Components with mean, std, weight, and interval boundaries.
    """

    # Flatten all peak positions into one long DataFrame
    all_peaks_positions = pd.DataFrame()
    for patient, patient_peaks in all_peaks.items():
        peaks_positions = pd.DataFrame(patient_peaks['Positions'], columns=['Peaks'])
        peaks_positions.index = [f"{patient}_peak_{i}" for i in range(1, len(peaks_positions) + 1)]
        all_peaks_positions = pd.concat([all_peaks_positions, peaks_positions])

    # Try multiple GMM fits and evaluate with BIC
    bic_scores = {}
    gm_models = {}

    for n in range(2, max_components + 1):
        gm = GaussianMixture(n_components=n, covariance_type='full', random_state=n)
        gm.fit(all_peaks_positions.values)

        bic = gm.bic(all_peaks_positions.values)
        bic_scores[n] = bic
        gm_models[n] = gm

        if show_plots:
            x = np.linspace(all_peaks_positions.min() - 10, all_peaks_positions.max() + 10, 1000).reshape(-1, 1)
            logprob = gm.score_samples(x)
            pdf = np.exp(logprob)

            plt.hist(all_peaks_positions, bins=100, density=True, alpha=0.5, color='gray')
            plt.plot(x, pdf, color='red', lw=2)
            plt.xlabel('Raman shift (cm^-1)')
            plt.ylabel('Density')
            plt.title(f'GMM with {n} components, BIC = {bic:.2f}')
            plt.show()

    # Select model with lowest BIC
    best_n = min(bic_scores, key=bic_scores.get)
    best_gm = gm_models[best_n]
    print(f"The best model has {best_n} components")

    # Extract GMM parameters
    means = best_gm.means_.flatten()
    stds = np.sqrt(best_gm.covariances_.flatten())
    weights = best_gm.weights_.flatten()

    # Define intervals as [mu - sigma, mu + sigma]
    intervals = np.column_stack([means - stds, means + stds])

    common_peaks = pd.DataFrame({
        'Means': means,
        'Stds': stds,
        'Weights': weights,
        'Min_Int': intervals[:, 0],
        'Max_Int': intervals[:, 1]
    }, index=[f"component_{i}" for i in range(1, best_n + 1)])

    # Sort components by mean position
    common_peaks = common_peaks.sort_values(by='Means')

    return common_peaks


# ==============================================================
# 3) Compute prominence of each interval in each spectrum
# ==============================================================
def compute_prominences(all_normalized_spectra, common_peaks, common_grid):
    """
    For each patient spectrum, compute the maximum intensity (prominence)
    within each interval defined by GMM components.

    Parameters
    ----------
    all_normalized_spectra : DataFrame
        Normalized spectra with common grid.
    common_peaks : DataFrame
        GMM components with intervals.
    common_grid : ndarray
        Common wavenumber grid.

    Returns
    -------
    prominences : DataFrame
        Each row = patient, each column = P1...Pn interval maxima.
    """

    common_grid = pd.Series(common_grid, name='Wavenumbers')
    prominences = pd.DataFrame()

    for patient in all_normalized_spectra.index:
        spectrum = all_normalized_spectra.loc[patient].to_frame(name=patient)
        spectrum['Wavenumbers'] = common_grid

        prominences_patient = []
        for component in common_peaks.index:
            min_int = common_peaks.loc[component, 'Min_Int']
            max_int = common_peaks.loc[component, 'Max_Int']

            # Select portion of spectrum in this interval
            spectrum_chunk = spectrum[(spectrum['Wavenumbers'] >= min_int) & (spectrum['Wavenumbers'] < max_int)]
            prominence_int = spectrum_chunk[patient].max()
            prominences_patient.append(prominence_int)

        # Store as one row
        prominences_patient = pd.DataFrame([prominences_patient], index=[patient],
                                           columns=[f"P{i}" for i in range(1, common_peaks.shape[0] + 1)])
        prominences = pd.concat([prominences, prominences_patient])

    return prominences


# ==============================================================
# 4) Build feature matrix and labels
# ==============================================================
def create_labels_and_dataset(prominences):
    """
    Create raw ratio dataset and binary labels from prominence data.

    Labels:
        - If patient name contains '+Ag' → 1
        - Else → 0

    Parameters
    ----------
    prominences : DataFrame
        Interval prominences per patient.

    Returns
    -------
    data : DataFrame
        Raw ratio dataset (all Pi/Pj and Pj/Pi, no NaN if possible).
    labels : Series
        Binary labels.
    """
    labels = pd.Series(
        [1 if "+Ag" in patient else 0 for patient in prominences.index],
        name="Label",
        index=prominences.index
    )

    ratio_dicts = []
    for patient in prominences.index:
        ratios_patient = {}
        for i in range(1, prominences.shape[1] + 1):
            val_i = prominences.loc[patient, f"P{i}"]
            for j in range(1, prominences.shape[1] + 1):
                if i == j:
                    continue
                val_j = prominences.loc[patient, f"P{j}"]

                # Both directions, avoid division by zero
                ratios_patient[f"P{i}/P{j}"] = val_i / val_j if val_j != 0 else np.nan
                ratios_patient[f"P{j}/P{i}"] = val_j / val_i if val_i != 0 else np.nan

        ratio_dicts.append(ratios_patient)

    data = pd.DataFrame(ratio_dicts, index=prominences.index)
    # # Fill NaN with median to fully match original behavior
    # data = data.fillna(data.median())

    return data, labels


# ==============================================================
# 5) Select best ratios and fill NaN only at the end
# ==============================================================
def select_ratios(data):
    """
    Select the ratio with higher relative variability (std/mean) from each pair of inverse ratios (Pi/Pj vs Pj/Pi).
    NaN values are filled with median only after selection.

    Parameters
    ----------
    data : DataFrame
        Raw ratio dataset (all Pi/Pj and Pj/Pi, NaN possible).

    Returns
    -------
    data_selected : DataFrame
        Dataset with only the ratios with higher relative variability, NaN filled.
    """
    pairs, visited = [], set()
    for f in data.columns:
        if f not in visited:
            num, den = f.split("/")
            inverse = f"{den}/{num}"
            if inverse in data.columns:
                pairs.append((f, inverse))
                visited.update([f, inverse])

    # Keep the ratio with higher relative variability (std/mean)
    to_keep = []
    for ratio1, ratio2 in pairs:
        r1, r2 = data[ratio1], data[ratio2]
        score1 = r1.std(skipna=True) / r1.mean(skipna=True)
        score2 = r2.std(skipna=True) / r2.mean(skipna=True)
        if score1 >= score2:
            to_keep.append(ratio1)
        else:
            to_keep.append(ratio2)

    # Keep only selected ratios and fill missing values with median
    data_selected = data[to_keep].copy()
    data_selected = data_selected.fillna(data_selected.median())
    return data_selected


#%% Load files

# Project directory
paths = ProjectPaths("Kell")

# Raw and processed datasets directories
raw_path, proc_path = paths.get_datasets_path(processed='both')

# Subfolder containing the pretreated files
pretreated_folder = Path(raw_path) / "dati pretrattati"

# Find all .txt files in the subfolder
spectrum_files = list(pretreated_folder.glob("*.txt"))

# Dictionary to store all spectra: key = file name, value = DataFrame with 'Wave' and 'Intensity'
spectra_dict = {}

for file_path in spectrum_files:
    # File name without extension
    spectrum_name = file_path.stem
    # Remove "_sub" at the end of the patient spectrum name
    spectrum_name = spectrum_name.removesuffix("_sub")
    print("Loading spectrum:", spectrum_name)

    # Load the file
    df = pd.read_csv(file_path, sep="\t")

    # Handle files with either 2 columns or a single combined column
    if df.shape[1] == 2:
        df.columns = ["Wave", "Intensity"]
    else:
        # Split the single column into two
        df = df[0].str.split("\t", expand=True)
        df.columns = ["Wave", "Intensity"]

    # Convert columns to float
    df = df.astype({"Wave": float, "Intensity": float})

    # Store in the dictionary
    spectra_dict[spectrum_name] = df


del df, spectrum_name, file_path, spectrum_files, pretreated_folder


#%% Analysis pipeline

# Ask to show plots
show_plots = input("Do you want to show plots? [y/N]: ").strip().lower() == "y"

# 1) Preprocess spectra
all_normalized_spectra, all_peaks, common_grid = preprocess_spectra(spectra_dict, show_plots=show_plots)

# 2) Fit Gaussian Mixture Model (GMM) on detected peaks
common_peaks = fit_gmm(all_peaks, show_plots=show_plots)

# 3) Compute prominences (interval maxima) for each spectrum
prominences = compute_prominences(all_normalized_spectra, common_peaks, common_grid)

# 4) Build feature matrix and extract labels
raw_data, labels = create_labels_and_dataset(prominences)

# 5) Select best ratios
proc_data = select_ratios(raw_data)


#%% Save dataframes

# Concatenate df with labels
df_final = pd.concat([proc_data, labels], axis=1)

# Save complete df
df_final.to_parquet(proc_path / "data.parquet", index=True)  
