# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 09:22:42 2025

@author: mik16
"""

#%% Libraries and submodules

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from sklearn.mixture import GaussianMixture
    
# Submodules utilities
from Utils.project_paths import ProjectPaths


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


#%%

def preprocess_spectrum(wn, intensity, common_grid, window_length=91, polyorder=3):
    """
    Interpolate, smooth, clip, and normalize a spectrum.

    Parameters
    ----------
    wn : ndarray
        Original wavenumbers.
    intensity : ndarray
        Original intensity values.
    common_grid : ndarray
        Target wavenumber grid for interpolation.
    window_length : int, optional
        Window length for Savitzky-Golay smoothing. Must be odd.
    polyorder : int, optional
        Polynomial order for Savitzky-Golay smoothing.

    Returns
    -------
    full_spectrum : ndarray
        Spectrum interpolated, smoothed, clipped, normalized, and mapped onto
        the full common grid.
    grid_interp : ndarray
        Subset of the grid where interpolation was performed (within data range).
    smoothed : ndarray
        Smoothed intensity values on the interpolated grid.
    """
    # Limit interpolation to the range of original data
    mask = (common_grid >= wn.min()) & (common_grid <= wn.max())
    grid_interp = common_grid[mask]

    # Monotone spline interpolation (PCHIP)
    pchip = PchipInterpolator(wn, intensity)
    intensity_interp = pchip(grid_interp)

    # Savitzky-Golay smoothing
    smoothed = savgol_filter(intensity_interp, window_length=window_length, polyorder=polyorder)

    # Clip negative values
    smoothed[smoothed < 0] = 0

    # Normalize area under curve to 1
    intensity_norm = smoothed / smoothed.sum()

    # Fill full spectrum across the entire grid (zeros outside original range)
    full_spectrum = np.zeros_like(common_grid, dtype=float)
    full_spectrum[mask] = intensity_norm

    return full_spectrum, grid_interp, smoothed


def detect_peaks(spectrum, grid, prominence=0.0001):
    """
    Detect peaks in a normalized spectrum.

    Parameters
    ----------
    spectrum : ndarray
        Full normalized spectrum on the common grid.
    grid : ndarray
        Common wavenumber grid.
    prominence : float, optional
        Minimum prominence for peak detection.

    Returns
    -------
    peaks_positions : ndarray
        Wavenumber positions of detected peaks.
    peaks_heights : ndarray
        Heights of detected peaks.
    """
    peaks, _ = find_peaks(spectrum, prominence=prominence)
    peaks_positions = grid[peaks]
    peaks_heights = spectrum[peaks]
    return peaks_positions, peaks_heights


def plot_spectrum(patient, wn, intensity, grid_interp, intensity_interp,
                  smoothed, common_grid, full_spectrum,
                  peaks_positions, peaks_heights):
    """
    Plot the preprocessing steps of a spectrum in four panels.

    Parameters
    ----------
    patient : str
        Patient identifier.
    wn : ndarray
        Original wavenumbers.
    intensity : ndarray
        Original intensity values.
    grid_interp : ndarray
        Interpolated grid.
    intensity_interp : ndarray
        Interpolated intensity values.
    smoothed : ndarray
        Smoothed intensity values.
    common_grid : ndarray
        Full common grid.
    full_spectrum : ndarray
        Final normalized spectrum on the common grid.
    peaks_positions : ndarray
        Detected peak positions.
    peaks_heights : ndarray
        Detected peak heights.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(wn, intensity, color='blue')
    axs[0, 0].set_title('Preprocessed')

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
    

#%% Parameters

# Define common wavenumber grid (all spectra will have same length)
min_wn = 200
max_wn = 2000
step = 1
common_grid = np.arange(min_wn, max_wn + 1, step)

# Savitzky-Golay parameters
window_length = 91
polyorder = 3

# Peak detection threshold
peak_threshold = 0.0001

# Store all normalized spectra
all_normalized_spectra = pd.DataFrame()

# Dictionary: key = patient, value = dict with 'Positions' and 'Heights'
all_peaks = {}


#%% Main loop: preprocess spectra and detect peaks

spectra_results = {}  # store results per patient

for patient, data in spectra_dict.items():
    wn = data['Wave'].values
    intensity = data['Intensity'].values

    full_spectrum, grid_interp, smoothed = preprocess_spectrum(
        wn, intensity, common_grid,
        window_length=window_length, polyorder=polyorder
    )

    peaks_positions, peaks_heights = detect_peaks(full_spectrum, common_grid, prominence=peak_threshold)

    spectra_results[patient] = {
        "wn": wn,
        "intensity": intensity,
        "grid_interp": grid_interp,
        "intensity_interp": PchipInterpolator(wn, intensity)(grid_interp),
        "smoothed": smoothed,
        "full_spectrum": full_spectrum,
        "peaks_positions": peaks_positions,
        "peaks_heights": peaks_heights
    }

    # Save normalized spectrum into DataFrame
    full_spectrum_df = pd.Series(full_spectrum).to_frame().T
    full_spectrum_df.index = [patient]
    all_normalized_spectra = pd.concat([all_normalized_spectra, full_spectrum_df], axis=0)

    # Store peaks
    all_peaks[patient] = {"Positions": peaks_positions, "Heights": peaks_heights}


#%% Plot all spectra (optional)

for patient, results in spectra_results.items():
    plot_spectrum(
        patient,
        results["wn"],
        results["intensity"],
        results["grid_interp"],
        results["intensity_interp"],
        results["smoothed"],
        common_grid,
        results["full_spectrum"],
        results["peaks_positions"],
        results["peaks_heights"]
    )


#%% 2) Interpolazione

# Definizione griglia comune
min_wn = 200
max_wn = 2000
step = 1
common_grid = np.arange(min_wn, max_wn + 1, step)  # tutti gli spettri avranno 1801 punti

# Parametri Savitzky-Golay
wl = 91
po = 3

# Threshold per picchi
peak_th = 0.0001

# Spettri normalizzati
all_normalized_spectra = pd.DataFrame()

# Mostra i plot
show_plot = False

# Dizionario con chiave = paziente e valore = dizionario con chiavi 'Posizioni' e 'Altezze' dei picchi trovati.
all_peaks = {}

for patient, data in spectra_dict.items():
    
    # Wavenumbers e Intensità del soggetto
    wn = data['Wave'].values
    intensity = data['Intensity'].values
    
    # Interpolazione spline monotona (PCHIP) sulla griglia comune
    # Limiti interpolazione ai dati originali per evitare overshoot agli estremi
    mask = (common_grid >= wn.min()) & (common_grid <= wn.max())
    grid_interp = common_grid[mask]
    
    pchip = PchipInterpolator(wn, intensity)
    intensity_interp = pchip(grid_interp)
    
    # Smoothing Savitzky-Golay su griglia uniforme
    smoothed = savgol_filter(intensity_interp, window_length=wl, polyorder=po)
    
    # Clip dei valori negativi
    smoothed[smoothed < 0] = 0
    
    # Normalizzazione area sotto curva = 1
    intensity_norm = smoothed / smoothed.sum()
    
    # Riempimento in array completo su tutta la griglia comune
    # Punti fuori dal range dei dati originali vengono messi a zero
    full_spectrum = np.zeros_like(common_grid, dtype=float)
    full_spectrum[mask] = intensity_norm
    
    full_spectrum_df = pd.Series(full_spectrum).to_frame().T
    full_spectrum_df.index = [patient]
    
    all_normalized_spectra = pd.concat([all_normalized_spectra, full_spectrum_df], axis = 0)
    
    # Rilevazione picchi
    peaks, _ = find_peaks(full_spectrum, prominence = peak_th)
    peaks_positions = common_grid[peaks]
    peaks_heights = full_spectrum[peaks]
    
    peaks_dict = {'Positions': peaks_positions, 'Heights': peaks_heights}
    
    all_peaks[patient] = peaks_dict
    
    if show_plot:
    
        # Plots 4 panel
        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        
        axs[0,0].plot(wn, intensity, color='blue')
        axs[0,0].set_title('Preprocessed')
        
        axs[0,1].plot(grid_interp, intensity_interp, color='red')
        axs[0,1].set_title('Interpolated')
        
        axs[1,0].plot(grid_interp, smoothed, color='green')
        axs[1,0].set_title('Smoothed')
        
        axs[1,1].plot(common_grid, full_spectrum, color='brown')
        axs[1,1].plot(peaks_positions, peaks_heights, 'ro')  # evidenzia picchi
        axs[1,1].set_title('Normalized + Peaks')
        
        for ax in axs.flat:
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        
        plt.suptitle(patient)
        plt.tight_layout()
        plt.show()
        
        del ax, axs, fig
    
#     del data, full_spectrum, full_spectrum_df, grid_interp, wn, intensity, intensity_interp, intensity_norm, smoothed, mask, patient
#     del peaks, peaks_dict, peaks_heights, peaks_positions

# del min_wn, max_wn, step, wl, po, peak_th
# del show_plot
# del spectra_dict

#%% 3) Gaussian Mixture Model

# Concateno le posizioni di tutti i picchi di tutti i pazienti e faccio reshape perchè così serve all'algoritmo GMM di sklearn
all_peaks_positions = pd.DataFrame()

for patient, patient_peaks in all_peaks.items():
    
    peaks_positions = pd.DataFrame(patient_peaks['Positions'])
    peaks_positions.columns = ['Peaks']
    peaks_positions.index = [patient + f"_peak_{i}" for i in np.arange(1, peaks_positions.shape[0] + 1, 1)]
    
    all_peaks_positions = pd.concat([all_peaks_positions, peaks_positions], axis = 0)
    
    del patient, patient_peaks, peaks_positions

del all_peaks

# Selezione del numero ottimale di componenti usando BIC
bic_scores = {}
gm_models = {}
max_components = 19

for n in range(2, max_components + 1):
    
    # Definizione del modello GMM
    gm = GaussianMixture(n_components = n, covariance_type = 'full', random_state = n)
    
    # Fit del modello
    gm.fit(all_peaks_positions.values)
    bic = round(gm.bic(all_peaks_positions.values), 3)
    bic_scores[f"components_{n}"] = bic
    gm_models[f"components_{n}"] = gm
    
    # Plot
    x = np.linspace(all_peaks_positions.min()-10, all_peaks_positions.max()+10, 1000).reshape(-1,1)
    logprob = gm.score_samples(x)
    pdf = np.exp(logprob)
    
    plt.hist(all_peaks_positions, bins=100, density=True, alpha=0.5, color='gray')
    plt.plot(x, pdf, color='red', lw=2)
    plt.xlabel('Raman shift (cm^-1)')
    plt.ylabel('Density')
    plt.ylim((0, 0.006))
    plt.title(f'Univariate GMM fit with {n} components\nBIC = {bic}.')
    plt.show()
    
    del gm, bic, x, logprob, pdf
del n, max_components
    

# Numero ottimale di componenti
best_idx = min(bic_scores, key = bic_scores.get)
best_gm = gm_models[best_idx]
best_bic = round(best_gm.bic(all_peaks_positions.values), 3)
best_idx = int(best_idx[-2:])
print(f"Optimal number of components: {best_idx}")

del bic_scores, gm_models

# Visualizzazione
x = np.linspace(all_peaks_positions.min()-10, all_peaks_positions.max()+10, 1000).reshape(-1,1)
logprob = best_gm.score_samples(x)
pdf = np.exp(logprob)

plt.hist(all_peaks_positions, bins=100, density=True, alpha=0.5, color='gray')
plt.plot(x, pdf, color='blue', lw=2)
plt.xlabel('Raman shift (cm^-1)')
plt.ylabel('Density')
plt.ylim((0, 0.006))
plt.title(f'Best Univariate GMM fit with {best_idx} components.\nLowest BIC = {best_bic}.')
plt.show()

# del x, logprob, pdf, all_peaks_positions