# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 11:58:25 2025

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


#%%

class ParallelModelTrainer:
    
    # Definiamo la namedtuple per il risultato del fold
    FoldResult = namedtuple('FoldResult', ['fold_idx', 'val_idx', 'selected_features',
                                           'scaler', 'model', 'y_pred', 'y_pred_proba'])
    
    def __init__(self, X, y, rskf, scaler, model, balancer=None, 
                 feature_selectors=None, classi_da_salvare=None,
                 n_cores=-1):
        """
        Inizializza il trainer per la cross-validation parallela.

        :param X: DataFrame delle feature
        :param y: Series o array del target
        :param rskf: un oggetto di tipo StratifiedKFold o RepeatedStratifiedKFold
        :param scaler: un oggetto di scaler (come StandardScaler, MinMaxScaler, ecc.)
        :param model: un modello di machine learning da usare (default è None)
        :param feature_selectors: una lista di oggetti di selezione delle feature (default è None)
        """
        self.X = X
        self.y = y
        self.rskf = rskf
        self.scaler = scaler
        self.model = model
        self.balancer = balancer
        self.feature_selectors = feature_selectors or []
        self.classi_da_salvare = classi_da_salvare
        self.n_cores = n_cores

    def process_fold(self, fold_idx, train_idx, val_idx):
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
        
        # 1) Feature selection sul train originale
        for selector in self.feature_selectors:
            selector.set_output(transform='pandas')
            X_train = selector.fit_transform(X_train, y_train)
            X_val = selector.transform(X_val)
        
        selected_features = X_train.columns
        
        # 2) Oversampling su train con feature selezionate
        if self.balancer is not None:
            balancer = clone(self.balancer)
            X_train, y_train = balancer.fit_resample(X_train, y_train)
            # converti se necessario
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=selected_features)
        
        # 3) Scaling
        scaler = clone(self.scaler)
        model = clone(self.model)
        
        X_train_scaled = scaler.fit_transform(X_train, y_train)
        X_val_scaled = scaler.transform(X_val)
    
        # Fit modello
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train_scaled, y_train)
    
        # Predizioni
        y_pred = model.predict(X_val_scaled)
    
        # Probabilità
        if self.classi_da_salvare is None:
            y_pred_proba = None
        else:
            proba = model.predict_proba(X_val_scaled)
            if len(self.classi_da_salvare) == 1:
                y_pred_proba = proba[:, self.classi_da_salvare[0]]
            else:
                y_pred_proba = proba[:, self.classi_da_salvare]
    
        # Restituiamo un oggetto di tipo FoldResult
        return self.FoldResult(fold_idx, val_idx, selected_features, scaler, model, y_pred, y_pred_proba)
    
    def parallel_training(self):
        n_splits = self.rskf.get_n_splits(self.X, self.y)

        # Parallelizza l'esecuzione del training su tutti i fold
        results = Parallel(n_jobs=self.n_cores, backend="loky", verbose=10)(
            delayed(self.process_fold)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in tqdm(enumerate(self.rskf.split(self.X, self.y)), total=n_splits)
        )

        return results

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
        """
        Restituisce un DataFrame con le predizioni dei fold in cross-validation.
        La prima colonna "Label" contiene i valori originali di y.
        """
        # Creiamo un dizionario con le chiavi uguali agli indici della Series y
        # Ogni chiave conterrà la lista delle predizioni corrispondenti a quell'indice
        predictions_dict = {idx: [] for idx in self.y.index}
    
        # Iteriamo sui risultati di ciascun fold
        for result in results:
            # result.val_idx contiene gli indici originali della Series y per il fold di validazione
            for i, idx in enumerate(result.val_idx):
                # Inseriamo la predizione corrispondente al fold nel dict
                # idx è già un indice valido della Series y, non serve loc qui
                predictions_dict[idx].append(result.y_pred[i])
    
        # Creiamo il DataFrame dai dizionari, orientato per indice
        df_pred = pd.DataFrame.from_dict(predictions_dict, orient="index")
    
        # Inseriamo la colonna "Label" prendendo i valori originali di y
        # Qui usiamo .loc per allineare i valori della Series agli indici del DataFrame
        df_pred.insert(0, "Label", self.y.loc[df_pred.index])
    
        # Rinominiamo le colonne delle predizioni come Iter_1, Iter_2, ...
        df_pred.columns = ["Label"] + [f"Iter_{i + 1}" for i in range(df_pred.shape[1] - 1)]
    
        # Ordiniamo per indice (utile se l'ordine dei fold non è consecutivo)
        df_pred = df_pred.sort_index()
    
        return df_pred
    
    
    def get_predictions_proba(self, results):
        """
        Restituisce un DataFrame (o lista di DataFrame) con le probabilità predette dai fold.
        Funziona sia con una classe da salvare sia con più classi.
        """
        if not hasattr(self, "classi_da_salvare") or self.classi_da_salvare is None:
            return None
    
        classi_da_salvare = self.classi_da_salvare
    
        # Creiamo un dizionario annidato: 
        # chiavi = classi da salvare, valori = dict con chiavi = indici della Series y
        predictions_proba_dict = {
            class_idx: {idx: [] for idx in self.y.index} for class_idx in classi_da_salvare
        }
    
        # Iteriamo sui risultati dei fold
        for result in results:
            for i, idx in enumerate(result.val_idx):
                # idx è l'indice originale della Series y
                # Caso: 1 sola classe → scalare
                if isinstance(result.y_pred_proba[i], (float, np.floating)):
                    class_idx = classi_da_salvare[0]
                    predictions_proba_dict[class_idx][idx].append(result.y_pred_proba[i])
                else:
                    # Caso: più classi → result.y_pred_proba[i] è un array di probabilità
                    for class_idx in classi_da_salvare:
                        predictions_proba_dict[class_idx][idx].append(result.y_pred_proba[i][class_idx])
    
        # Convertiamo i dizionari in DataFrame
        df_pred_proba_list = []
        for class_idx in classi_da_salvare:
            df_class = pd.DataFrame.from_dict(predictions_proba_dict[class_idx], orient="index")
            # Inseriamo la colonna "Label" usando .loc per allineare con gli indici
            df_class.insert(0, "Label", self.y.loc[df_class.index])
            df_class.columns = ["Label"] + [f"Iter_{i + 1}" for i in range(df_class.shape[1] - 1)]
            df_class = df_class.sort_index()
            df_pred_proba_list.append(df_class)
    
        # Se c'è solo una classe, restituiamo un singolo DataFrame
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