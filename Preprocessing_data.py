# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 09:22:42 2025

@author: mik16
"""

#%% Base libraries and project root

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


#%% Third-party libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from sklearn.mixture import GaussianMixture


#%% Local / project submodules

from Utils.project_paths import ProjectPaths
# from ML_Submodule import ml_model  # se vuoi aggiungere altri submodules


#%% Load files

# Imposta la directory di lavoro
paths = ProjectPaths("Kell")

# Imposta le directory dei dataset
raw_path, proc_path = paths.get_datasets_path(processed='both')

