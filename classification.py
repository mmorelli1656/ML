# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:42:29 2025

@author: mik16
"""

#%% Libraries and submodules

import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib
from joblib import cpu_count
from pathlib import Path
    
# Submodules utilities
from Utils.project_paths import ProjectPaths
from ML.my_featsel import FeaturesVariance, FeaturesPearson
from ML.my_parallel_ML import ParallelModelTrainer
from ML.my_eval import EvaluationMetrics

sys.path.append(r"C:\Users\mik16\Github\Kell")  # cartella che contiene ML


#%% Functions

def variance_histogram(X, bins, percentile=None):
    """
    Applica Min-Max scaling al DataFrame X, calcola la varianza per colonna
    e mostra un istogramma della distribuzione delle varianze.
    
    Se specificato, aggiunge una linea verticale per il valore di un percentile.

    Parametri:
        X: DataFrame originale
        bins: numero di intervalli dell'istogramma
        percentile: valore percentile da evidenziare (0-100) oppure None

    Ritorna:
        X_scaled: DataFrame scalato
        variances: Series con varianze delle colonne
    """
    # Copia e scala
    X_scaled = X.copy()
    scaler = MinMaxScaler()
    X_scaled[:] = scaler.fit_transform(X_scaled)

    # Calcola varianza
    variances = X_scaled.var()

    # Plot istogramma
    plt.figure(figsize=(10, 6))
    plt.hist(variances, bins=bins, color='skyblue', edgecolor='black')

    # Percentile opzionale
    if percentile is not None and 0 <= percentile <= 100:
        perc_value = np.percentile(variances, percentile)
        plt.axvline(perc_value, color='red', linestyle='--', linewidth=2,
                    label=f'{percentile}° percentile = {perc_value:.4f}')
        plt.legend(fontsize=12)

    plt.title('Distribuzione delle varianze', fontsize=16)
    plt.xlabel('Varianza', fontsize=14)
    plt.ylabel('N° features', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return variances


#%% Load files

# Project directory
paths = ProjectPaths("Kell")

# Processed datasets directory
proc_path = paths.get_datasets_path(processed=True)

# Load dataframe
X = pd.read_parquet(proc_path / "data.parquet")

# Extract label
y = X.pop("Truth")

del proc_path


#%% Istogramma delle varianze

_ = variance_histogram(X, bins='auto', percentile=90)


#%% Pipeline

# Inizializza la CV
rskf = RepeatedStratifiedKFold(
    n_splits=10,        # numero di fold
    n_repeats=100,       # numero ripetizioni
    random_state=42   # riproducibilità
)

# Inizializza i metodi di FS
feature_selectors = [FeaturesVariance(threshold_value=90, mode='percentile'),
                     FeaturesPearson(threshold=0.9, alpha=0.01, random_state=42)]

# Inizializza lo scaler
scaler = StandardScaler()

# Modello base con parametri fissi
model = xgb.XGBClassifier(
    eval_metric='logloss',
    n_jobs=-1,
    early_stopping_rounds=20,
    random_state=42
)

# Parametri XGB per 60% con M>=5
# model = xgb.XGBClassifier(
#     eval_metric='logloss',
#     n_jobs=-1,
#     random_state=42,
#     early_stopping_rounds=20,
#     colsample_bytree=0.60,      # quantità di feature da campionare per ogni albero
#     # gamma=1,                   # regolarizzazione minima per uno split
#     # learning_rate=0.1,         # alias: eta
#     max_depth=3,               # profondità massima degli alberi
#     n_estimators=100,          # numero di boosting rounds
#     subsample=0.60             # percentuale di campionamento dei dati per albero
# )


# Inizializza il modello
# model = LogisticRegression(max_iter=200)  # aumento max_iter se serve


#%% Addestramento e risultati

print(f"Numero di core logici disponibili: {cpu_count()}")

# Inizializza l'addestramento parallelo
trainer = ParallelModelTrainer(X, y, rskf, scaler, model, balancer=None,
                               feature_selectors=None,
                               classi_da_salvare=[0,1])

# Addestra in parallelo
results = trainer.parallel_training()

# Estrai i risultati
df_feat_sel = trainer.get_feature_selection(results)
df_pred = trainer.get_predictions(results)
list_df_pred_proba = trainer.get_predictions_proba(results)
# list_scaler_model = trainer.get_scaler_model(results)

# Estrai tutti i risultati
final_results = trainer.get_all(results)

# Salvataggio risultati
# save_results(final_results, results_path, save_scalers=False, save_models=False)

# Conta quante features vengono prese in ogni split
columns_with_ones_dict = {
    idx: list(df_feat_sel.columns[row == 1])
    for idx, row in df_feat_sel.iterrows()
}

del rskf, feature_selectors, scaler, model, trainer


#%% Valutazione dei modelli 

# image_path = None

# Estrai le probabilità della classe di interesse
df_pred_proba = list_df_pred_proba[1]

# Inizializza l'evaluator per le performance del modello
evaluator = EvaluationMetrics(df_pred=df_pred, df_pred_proba=df_pred_proba,
                              task="binaryclass")

# Performances
df_metrics = evaluator.compute_metrics()

# Plot della CM
classes_name = ['Non-Earthquake', 'Earthquake']
evaluator.plot_confusion_matrix(perc='row', stat_method="mean_std",
                                classes=classes_name, save_path=image_path)

# Plot ROCs
evaluator.plot_roc_curve(stat_method="mean_std", save_path=image_path)

# Plot boxplots delle metriche
evaluator.plot_metrics_boxplot(df_metrics, save_path=image_path)

del classes_name, df_pred_proba, evaluator