# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 12:22:39 2025

@author: mik16
"""

#%%

import sys
import os
import pandas as pd
import numpy as np

# Definisci il percorso della cartella con il modulo
directory = r"C:\Users\mik16\OneDrive - Università degli Studi di Bari (1)\Projects\THOR"

# Cambia la directory di lavoro
os.chdir(directory)

my_ML_path = r"C:\Users\mik16\OneDrive - Università degli Studi di Bari (1)\Projects\Altro\01_Codes"
sys.path.append(my_ML_path)

# Importo il modulo
from my_eval import EvaluationMetrics

del my_ML_path, directory


#%%

# Per risultati riproducibili
np.random.seed(42)  

# Esempio di df per classificazione binaria
df_binaryclass = pd.DataFrame({
    'true_labels': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'pred1': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    'pred2': [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    'pred3': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# Esempio di probabilità per classificazione binaria
df_binaryclass_proba = pd.DataFrame({
    'true_labels': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'pred1': np.clip(np.random.normal(0.8, 0.1, 10), 0, 1),  
    'pred2': np.clip(np.random.normal(0.6, 0.15, 10), 0, 1),
    'pred3': np.clip(np.random.normal(0.4, 0.2, 10), 0, 1),  
})

# Esempio di df per classificazione multiclasse
df_multiclass = pd.DataFrame({
    'true_labels':    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    'pred1':          [0, 1, 2, 0, 1, 1, 2, 1, 2, 0],
    'pred2':          [0, 2, 2, 1, 0, 2, 1, 1, 2, 0],
    'pred3':          [1, 1, 2, 0, 2, 2, 0, 1, 1, 0]
})

# Esempio di df per regressione
true_labels = np.random.uniform(0, 100, 10)

df_regression = pd.DataFrame({
    'true_labels': true_labels,
    'pred1': true_labels + np.random.normal(0, 5, 10),
    'pred2': true_labels + np.random.normal(0, 8, 10),
    'pred3': true_labels + np.random.normal(0, 3, 10),
})

del true_labels


#%% Evaluator classificazione binaria

evaluator = EvaluationMetrics(df_pred=df_binaryclass, df_pred_proba=df_binaryclass_proba,
                              task="binaryclass")
df_metrics = evaluator.compute_metrics()
classes_name = ['Sano', 'Malato']
# evaluator.plot_confusion_matrix(perc='row', classes=classes_name, save=False)
# evaluator.plot_roc_curve(save=False)
evaluator.plot_metrics_boxplot(df_metrics, save=False)

del classes_name


#%% Evaluator classificazione multiclasse

evaluator = EvaluationMetrics(df_multiclass, df_pred_proba=None,
                              task="multiclass")
df_metrics = evaluator.compute_metrics()
classes_name = ['Sano', 'Malato', 'Dubbio']
evaluator.plot_confusion_matrix(perc='row', classes=classes_name, save=False)

del classes_name


#%% Evaluator regressione

evaluator = EvaluationMetrics(df_regression, df_pred_proba=None, task="regression")
df_metrics = evaluator.compute_metrics()
evaluator.plot_metrics_boxplot(df_metrics, save=False)
