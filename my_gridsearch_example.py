# -*- coding: utf-8 -*-
"""
Created on Sun May 25 08:56:57 2025

@author: mik16
"""

#%%

import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

my_ML_path = r"C:\Users\mik16\Github\ML"
sys.path.append(my_ML_path)

# Importo il modulo
from my_gridsearch import ParallelGridSearch

del my_ML_path


#%%

# Carica un dataset di esempio
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


#%%

# Inizializza i componenti
scaler = StandardScaler()
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 8, 10]
}
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Inizializza il tuo oggetto di ricerca
pgs = ParallelGridSearch(
    X=X,
    y=y,
    rskf=rskf,
    scaler=scaler,
    model=model,
    param_grid=param_grid,
    balancer=None,
    feature_selectors=[],  # puoi testare con selettori anche
    classi_da_salvare=[1],  # per AUC, classe positiva
    n_cores=-1  # usa tutti i core
)


#%%

# Esegui la training parallela
results = pgs.parallel_training()


#%%

# Aggrega i risultati con una metrica
summary_df = pgs.aggregate_results(results, metric_fn=roc_auc_score, use_proba=True)
summary_df_paral = pgs.aggregate_results_parallel(results, metric_fn=roc_auc_score, use_proba=True)


#%%

# Mostra i migliori parametri e punteggio
best_params = pgs.get_best_params(summary_df)
best_score = pgs.get_best_score(summary_df)

# Mostra il DataFrame dei risultati
print(summary_df)
