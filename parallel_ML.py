# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 19:06:46 2025

@author: mik16
"""

#%%

import pandas as pd
import numpy as np
import time
from collections import namedtuple
from sklearn.base import clone
from joblib import Parallel, delayed
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor


#%% Funzione process_fold esterna

def process_fold(X, y, train_idx, val_idx, scaler, model, balancer,
                 feature_selectors, classi_da_salvare, fold_idx):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Feature selection
    for selector in feature_selectors:
        selector.set_output(transform='pandas')
        X_train = selector.fit_transform(X_train, y_train)
        X_val = selector.transform(X_val)
    
    selected_features = X_train.columns
    
    # Oversampling
    if balancer is not None:
        bal = clone(balancer)
        X_train, y_train = bal.fit_resample(X_train, y_train)
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=selected_features)
    
    # Scaling e modello
    scaler_cloned = clone(scaler)
    model_cloned = clone(model)
    
    X_train_scaled = scaler_cloned.fit_transform(X_train, y_train)
    X_val_scaled = scaler_cloned.transform(X_val)
    
    # Fit modello
    if isinstance(model_cloned, (XGBClassifier, XGBRegressor)):
        model_cloned.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    else:
        model_cloned.fit(X_train_scaled, y_train)
    
    # Predizioni
    y_pred = model_cloned.predict(X_val_scaled)
    
    # Probabilità
    if classi_da_salvare is None:
        y_pred_proba = None
    else:
        proba = model_cloned.predict_proba(X_val_scaled)
        if len(classi_da_salvare) == 1:
            y_pred_proba = proba[:, classi_da_salvare[0]]
        else:
            y_pred_proba = proba[:, classi_da_salvare]
    
    # Restituisco come namedtuple
    FoldResult = namedtuple('FoldResult', ['fold_idx', 'val_idx', 'selected_features',
                                           'scaler', 'model', 'y_pred', 'y_pred_proba'])
    return FoldResult(fold_idx, val_idx, selected_features, scaler_cloned, model_cloned, y_pred, y_pred_proba)


#%% Classe ParallelModelTrainer

class ParallelModelTrainer:
    
    def __init__(self, X, y, rskf, scaler, model, balancer=None, 
                 feature_selectors=None, classi_da_salvare=None,
                 n_cores=-1):
        self.X = X
        self.y = y
        self.rskf = rskf
        self.scaler = scaler
        self.model = model
        self.balancer = balancer
        self.feature_selectors = feature_selectors or []
        self.classi_da_salvare = classi_da_salvare
        self.n_cores = n_cores

    def parallel_training(self):
        n_splits = self.rskf.get_n_splits(self.X, self.y)
        start_time = time.time()

        results = Parallel(n_jobs=self.n_cores, backend="loky")(
            delayed(process_fold)(
                self.X, self.y, train_idx, val_idx,
                self.scaler, self.model,
                self.balancer, self.feature_selectors,
                self.classi_da_salvare, fold_idx
            )
            for fold_idx, (train_idx, val_idx) in tqdm(enumerate(self.rskf.split(self.X, self.y)), total=n_splits)
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        if hours > 0:
            print(f"Training time: {hours} h, {minutes} min e {seconds:.2f} sec")
        elif minutes > 0:
            print(f"Training time: {minutes} min e {seconds:.2f} sec")
        else:
            print(f"Training time: {seconds:.2f} sec")

        return results

    # Tutti i metodi per estrarre feature selection, predizioni e modelli rimangono uguali
    # Esempio:
    def get_feature_selection(self, results):
        n_repeats = int(self.rskf.n_repeats)
        n_folds = int(self.rskf.get_n_splits(self.X, self.y) / n_repeats)
    
        row_names = [f"Iter_{r}_Fold_{f}"
                     for r in range(1, n_repeats + 1)
                     for f in range(1, n_folds + 1)]
        column_names = list(self.X.columns)
    
        df_feature_sel = pd.DataFrame(data=np.zeros((len(row_names), len(column_names))), columns=column_names)
    
        for result in results:
            df_feature_sel.loc[result.fold_idx, result.selected_features] = 1
    
        df_feature_sel.index = row_names
        return df_feature_sel

    def get_predictions(self, results):
        predictions_dict = {i: [] for i in range(len(self.y))}
    
        for result in results:
            for i, idx in enumerate(result.val_idx):
                predictions_dict[idx].append(result.y_pred[i])
    
        df_pred = pd.DataFrame.from_dict(predictions_dict, orient="index")
        df_pred.insert(0, "Label", self.y)
        df_pred.columns = ["Label"] + [f"Iter_{i + 1}" for i in range(df_pred.shape[1] - 1)]
        df_pred = df_pred.sort_index()
    
        return df_pred

    def get_predictions_proba(self, results):
        if not hasattr(self, "classi_da_salvare") or self.classi_da_salvare is None:
            return None
    
        classi_da_salvare = self.classi_da_salvare
        predictions_proba_dict = {
            class_idx: {i: [] for i in range(len(self.y))}
            for class_idx in classi_da_salvare
        }
    
        for result in results:
            for i, idx in enumerate(result.val_idx):
                if isinstance(result.y_pred_proba[i], (float, np.floating)):  
                    class_idx = classi_da_salvare[0]  
                    predictions_proba_dict[class_idx][idx].append(result.y_pred_proba[i])
                else:  
                    for class_idx in classi_da_salvare:
                        predictions_proba_dict[class_idx][idx].append(result.y_pred_proba[i][class_idx])
    
        df_pred_proba_list = []
        for class_idx in classi_da_salvare:
            df_class = pd.DataFrame.from_dict(predictions_proba_dict[class_idx], orient="index")
            df_class.insert(0, "Label", self.y)
            df_class.columns = ["Label"] + [f"Iter_{i + 1}" for i in range(df_class.shape[1] - 1)]
            df_class = df_class.sort_index()
            df_pred_proba_list.append(df_class)
    
        return df_pred_proba_list[0] if len(classi_da_salvare) == 1 else df_pred_proba_list
    
    def get_scaler_model(self, results):
        return [(result.scaler, result.model) for result in results]
    
    def get_all(self, results):
        return {
            "pred": self.get_predictions(results),
            "pred_proba": self.get_predictions_proba(results),
            "feature_sel": self.get_feature_selection(results),
            "scaler_model": self.get_scaler_model(results)
        }
