# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 09:26:47 2025

@author: mik16
"""

#%%

from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from collections import namedtuple, defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


#%%

class ParallelGridSearch:
    FoldResult = namedtuple('FoldResult', [
        'fold_idx', 'val_idx', 'selected_features',
        'scaler', 'param_combination', 'model',
        'y_pred', 'y_pred_proba'
    ])

    def __init__(self, X, y, rskf, scaler, model, param_grid, balancer=None,
                 feature_selectors=None, classi_da_salvare=None, n_cores=-1):
        """
        Inizializza il trainer per la cross-validation parallela.

        :param X: DataFrame delle feature
        :param y: Series o array del target
        :param rskf: oggetto di tipo StratifiedKFold o RepeatedStratifiedKFold
        :param scaler: oggetto scaler (es. StandardScaler, MinMaxScaler, ecc.)
        :param model: modello di machine learning da utilizzare
        :param param_grid: dizionario dei parametri per la grid search
        :param feature_selectors: lista di oggetti per la selezione delle feature
        :param classi_da_salvare: classi per le quali salvare le probabilità
        :param n_cores: numero di core da utilizzare per il parallelismo
        """
        self.X = X
        self.y = y
        self.rskf = rskf
        self.scaler = scaler
        self.model = model
        self.balancer = balancer
        self.param_grid = list(ParameterGrid(param_grid))
        self.feature_selectors = feature_selectors or []
        self.classi_da_salvare = classi_da_salvare
        self.n_cores = n_cores

    def process_fold(self, fold_idx, train_idx, val_idx):
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y[train_idx], self.y[val_idx]
        
        # Bilanciamento del dataset
        if self.balancer is not None:
            balancer = clone(self.balancer)
            X_train, y_train = balancer.fit_resample(X_train, y_train)
            # Se X_train è ora un ndarray, converti in DataFrame per compatibilità
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=self.X.columns)

        # Applica i selettori di feature
        for selector in self.feature_selectors:
            selector.set_output(transform='pandas')
            X_train = selector.fit_transform(X_train, y_train)
            X_val = selector.transform(X_val)

        selected_features = X_train.columns

        # Clona e applica lo scaler una sola volta per lo split
        scaler = clone(self.scaler)
        X_train_scaled = scaler.fit_transform(X_train, y_train)
        X_val_scaled = scaler.transform(X_val)

        results = []

        # Ciclo su tutte le combinazioni di parametri
        for param_comb in self.param_grid:
            model = clone(self.model)
            model.set_params(**param_comb)
            # Fit modello
            if isinstance(model, (XGBClassifier, XGBRegressor)):
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_val_scaled)

            if self.classi_da_salvare is None:
                y_pred_proba = None
            else:
                proba = model.predict_proba(X_val_scaled)
                if len(self.classi_da_salvare) == 1:
                    y_pred_proba = proba[:, self.classi_da_salvare[0]]
                else:
                    y_pred_proba = proba[:, self.classi_da_salvare]

            results.append(self.FoldResult(
                fold_idx=fold_idx,
                val_idx=val_idx,
                selected_features=selected_features,
                scaler=scaler,
                param_combination=param_comb,
                model=model,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba
            ))

        return results

    def parallel_training(self):
        n_splits = self.rskf.get_n_splits(self.X, self.y)

        all_results = Parallel(n_jobs=self.n_cores, backend="loky")(
            delayed(self.process_fold)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in tqdm(enumerate(self.rskf.split(self.X, self.y)), total=n_splits)
        )

        # Appiattisce la lista di liste
        return [item for sublist in all_results for item in sublist]


    def aggregate_results(self, results, metric_fn, use_proba=False):
        """
        Aggrega i risultati ottenuti da parallel_training, calcolando la metrica specificata
        (es. recall, AUC) per ogni combinazione di parametri.
    
        :param results: lista di FoldResult
        :param metric_fn: funzione di scoring (es. recall_score, roc_auc_score, ecc.)
        :param use_proba: se True, utilizza le probabilità invece delle etichette per calcolare la metrica
        :return: DataFrame con media e deviazione standard della metrica per combinazione di parametri
        """
        metric_scores = defaultdict(list)  # Dizionario per raccogliere i punteggi per ogni combinazione di parametri
    
        # Itera su ciascun risultato dei fold (risultati della cross-validation)
        for res in results:
            y_true = self.y[res.val_idx]  # Etichette vere per la validazione
            y_pred = res.y_pred          # Etichette predette dal modello
            y_pred_proba = res.y_pred_proba  # Probabilità predette dal modello (utili per AUC)
    
            # Se usiamo le probabilità (per esempio per l'AUC), calcoliamo la metrica con quelle
            if use_proba:
                if y_pred_proba is None:  # Salta questo fold se le probabilità non sono disponibili
                    continue
                # Calcola la metrica usando le probabilità
                score = metric_fn(y_true, y_pred_proba)
            else:
                # Se non usiamo le probabilità, usiamo le etichette predette
                score = metric_fn(y_true, y_pred)
    
            # Ordina i parametri per creare una tupla univoca che rappresenta la configurazione del modello
            param_tuple = tuple(sorted(res.param_combination.items()))
    
            # Aggiungi il punteggio alla lista dei punteggi per questa combinazione di parametri
            metric_scores[param_tuple].append(score)
    
        # Crea una lista per raccogliere i risultati aggregati
        summary = []
    
        # Per ogni combinazione di parametri (tupla) raccogliamo la media e la deviazione standard della metrica
        for param_tuple, scores in metric_scores.items():
            param_dict = dict(param_tuple)  # Converte la tupla dei parametri in un dizionario
            mean_score = np.mean(scores)    # Calcola la media dei punteggi per questa combinazione
            std_score = np.std(scores)      # Calcola la deviazione standard dei punteggi
    
            # Aggiungi media e deviazione standard al dizionario dei parametri
            param_dict.update({'mean_score': mean_score, 'std_score': std_score})
            summary.append(param_dict)  # Aggiungi i risultati alla lista
    
        # Crea un DataFrame dai risultati aggregati
        df_summary = pd.DataFrame(summary)
    
        # Ordina i risultati in base alla media del punteggio (decrescente) e resetta l'indice
        df_summary = df_summary.sort_values(by='mean_score', ascending=False).reset_index(drop=True)
    
        return df_summary

    def aggregate_results_parallel(self, results, metric_fn, use_proba=False, n_jobs=-1):
        """
        Aggrega i risultati in parallelo, calcolando la metrica specificata.
        """
    
        def process_result(res):
            y_true = self.y[res.val_idx]
            y_pred = res.y_pred
            y_pred_proba = res.y_pred_proba
    
            if use_proba:
                if y_pred_proba is None:
                    return None
                score = metric_fn(y_true, y_pred_proba)
            else:
                score = metric_fn(y_true, y_pred)
    
            param_tuple = tuple(sorted(res.param_combination.items()))
            return (param_tuple, score)
        
        # Parallelizza l'elaborazione dei risultati
        processed = Parallel(n_jobs=n_jobs)(
            delayed(process_result)(res) for res in tqdm(results)
        )
    
        # Filtro i None (fold saltati)
        processed = [p for p in processed if p is not None]
    
        # Costruisci dizionario aggregato
        metric_scores = defaultdict(list)
        for param_tuple, score in processed:
            metric_scores[param_tuple].append(score)
    
        # Aggrega media e std
        summary = []
        for param_tuple, scores in metric_scores.items():
            param_dict = dict(param_tuple)
            param_dict.update({
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })
            summary.append(param_dict)
    
        df_summary = pd.DataFrame(summary)
        df_summary = df_summary.sort_values(by='mean_score', ascending=False).reset_index(drop=True)
    
        return df_summary

    def get_best_params(self, df_summary):
        """
        Restituisce il miglior set di parametri (la combinazione che ha dato il miglior punteggio medio).
    
        :param df_summary: DataFrame con i risultati aggregati
        :return: Il dizionario dei migliori parametri
        """
        best_params = df_summary.iloc[0]  # Prendi la prima riga (la migliore) del DataFrame
        best_params_dict = best_params.drop(['mean_score', 'std_score']).to_dict()  # Rimuovi 'mean_score' e 'std_score'
        
        print(f"Best Parameters: {best_params_dict}")
        return best_params_dict
    
    def get_best_score(self, df_summary):
        """
        Restituisce il miglior punteggio medio ottenuto durante la ricerca.
    
        :param df_summary: DataFrame con i risultati aggregati
        :return: Il miglior punteggio medio
        """
        best_score = df_summary.iloc[0]['mean_score']  # Prendi il punteggio medio della prima riga (la migliore)
        
        print(f"Best Score: {best_score}")
        return best_score