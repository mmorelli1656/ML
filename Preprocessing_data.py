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


#%% Load all pretreated spectra from "dati pretrattati"

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
    df = pd.read_csv(file_path, sep="\t", header=None)

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


del df, spectrum_name, file_path, spectrum_files


#%% Interpolazione

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

# Dizionario con chiave = paziente e valore = dizionario con chiavi 'Posizioni' e 'Altezze' dei picchi trovati.
all_peaks = {}

for patient, data in global_dict.items():
    
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
    
    del ax, axs, data, fig, full_spectrum, full_spectrum_df, grid_interp, wn, intensity, intensity_interp, intensity_norm, smoothed, mask, patient
    del peaks, peaks_dict, peaks_heights, peaks_positions

del common_grid, min_wn, max_wn, step, wl, po, peak_th