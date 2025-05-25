# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 18:10:16 2025

@author: mik16
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:24:29 2025

@author: mik16
"""

#%%

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr


#%%

class FeaturesVariance(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_value=90, mode="percentile"):
        """
        Inizializza il selettore con due modalità di calcolo della soglia di varianza.
        
        :param threshold_value: Percentile (0-100) o percentuale della varianza massima (0-100).
                                Se 'mode' è 'percentile', il valore rappresenta il percentile della varianza
                                delle feature da mantenere. Se 'mode' è 'max_percentage', il valore rappresenta
                                la percentuale della varianza massima da mantenere.
        :param mode: Modalità di calcolo per la soglia di varianza. Può essere:
                     - 'percentile': seleziona un percentile specificato della varianza delle feature.
                     - 'max_percentage': seleziona una percentuale della varianza massima delle feature.
        """
        # Verifica che la modalità sia corretta
        if mode not in ["percentile", "max_percentage"]:
            raise ValueError(f"Il parametro mode deve essere 'percentile' o 'max_percentage'. Valore fornito: {mode}")
        
        # Verifica che il valore della soglia sia valido
        if not (0 < threshold_value < 100):
            raise ValueError("threshold_value deve essere strettamente compreso tra 0 e 100. Valore fornito: {}".format(threshold_value))
        
        self.threshold_value = threshold_value
        self.mode = mode
        self._output_format = 'default'  # Imposta il formato dell'output come array NumPy di default

    def set_output(self, *, transform=None):
        """
        Imposta il formato dell'output del metodo transform.
        
        :param transform: 'default' per array NumPy o 'pandas' per DataFrame. Se None, viene utilizzato il formato di default.
        :return: Restituisce l'oggetto stesso per un uso a catena del metodo.
        """
        # Verifica che il formato di output sia uno dei valori consentiti
        if transform not in [None, 'default', 'pandas']:
            raise ValueError(f"transform deve essere 'default', 'pandas' o None. Valore fornito: {transform}")
        
        # Imposta il formato di output
        self._output_format = 'default' if transform is None else transform
        return self

    def fit(self, X, y=None):
        """
        Calcola la varianza per ogni feature e seleziona le feature in base alla soglia specificata.

        :param X: DataFrame o array numpy contenente le features.
        :param y: Non utilizzato in questa fase, presente per coerenza con l'interfaccia di scikit-learn.
        :return: Restituisce l'oggetto stesso per un uso a catena del metodo.
        """
        # Verifica se X è un DataFrame
        self._is_dataframe = isinstance(X, pd.DataFrame)
        
        # Se X è un DataFrame, convertilo in array numpy per l'elaborazione
        X_values = X.values if self._is_dataframe else X
        
        # Controllo su eventuali NaN nei dati
        if np.isnan(X_values).any():
            print("Errore: sono presenti valori NaN nei dati forniti a 'fit'. Rimuoverli o imputarli prima.")
            raise ValueError("Valori NaN trovati in X.")

        # Normalizza le features per avere valori tra 0 e 1
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_values)
        
        # Calcola la varianza di ogni feature (colonna) del dataset
        variances = np.var(X_scaled, axis=0)

        # Calcola la soglia in base alla modalità specificata
        if self.mode == "percentile":
            # Calcola il percentile specificato della varianza
            threshold = np.percentile(variances, self.threshold_value)
        elif self.mode == "max_percentage":
            # Calcola una percentuale della varianza massima
            max_var = np.max(variances)
            threshold = (self.threshold_value / 100) * max_var

        # Imposta la soglia e le feature selezionate
        self.threshold_ = threshold
        self.selected_features_ = np.where(variances >= threshold)[0]

        # Salva i nomi delle feature se l'input è un DataFrame
        if self._is_dataframe:
            self.input_features_ = X.columns
        else:
            # Se l'input non è un DataFrame, usa nomi generici per le colonne
            self.input_features_ = [f"x{i}" for i in range(X.shape[1])]

        return self

    def transform(self, X):
        """
        Seleziona solo le feature che superano la soglia di varianza.

        :param X: DataFrame o array numpy contenente le features.
        :return: Le features selezionate in base alla varianza.
        """
        # Se X è un DataFrame, convertilo in array numpy per l'elaborazione
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        # Seleziona solo le colonne che sono state selezionate durante il fitting
        transformed = X_values[:, self.selected_features_]

        # Restituisce il formato di output richiesto
        if self._output_format == 'pandas':
            # Se il formato richiesto è pandas, restituisci un DataFrame con i nomi delle feature selezionate
            selected_names = self.get_feature_names_out(self.input_features_)
            return pd.DataFrame(transformed, columns=selected_names, index=getattr(X, 'index', None))
        
        # Altrimenti restituisci l'array NumPy
        return transformed

    def fit_transform(self, X, y=None):
        """
        Esegue fit e poi trasforma i dati in un solo passaggio.
        
        :param X: DataFrame o array numpy contenente le features.
        :param y: Non utilizzato in questa fase, presente per coerenza con l'interfaccia di scikit-learn.
        :return: Le features selezionate in base alla varianza.
        """
        # Esegue fit e trasforma in un solo passaggio
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Restituisce i nomi delle feature selezionate.

        :param input_features: Lista/array dei nomi originali.
        :return: I nomi delle feature selezionate.
        """
        # Se non vengono forniti i nomi delle feature, usa quelli memorizzati durante il fitting
        if input_features is None:
            input_features = getattr(self, 'input_features_', [f"x{i}" for i in range(len(self.selected_features_))])
        
        # Restituisce i nomi delle feature selezionate
        return np.array(input_features)[self.selected_features_]


class FeaturesPearson(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, alpha=0.05, random_state=None):
        """
        Inizializza il selettore di feature basato sulla correlazione di Pearson.

        :param threshold: Soglia di correlazione oltre la quale si eliminano le feature.
        :param alpha: Livello di significatività per il p-value della correlazione.
        :param random_state: Seed per la generazione casuale, garantisce riproducibilità.
        """
        self.threshold = threshold
        self.alpha = alpha
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._output_format = 'default'

    def set_output(self, *, transform=None):
        if transform not in [None, 'default', 'pandas']:
            raise ValueError(f"transform deve essere 'default', 'pandas' o None. Valore fornito: {transform}")
        self._output_format = 'default' if transform is None else transform
        return self

    def fit(self, X, y=None):
        self._is_dataframe = isinstance(X, pd.DataFrame)
        X_df = X.copy() if self._is_dataframe else pd.DataFrame(X)

        self.input_features_ = X_df.columns if self._is_dataframe else [f"x{i}" for i in range(X_df.shape[1])]
        n_features = X_df.shape[1]
        to_remove = set()

        corr_matrix = X_df.corr().abs()
        upper_tri_indices = np.triu_indices(n_features, k=1)

        for i, j in zip(*upper_tri_indices):
            if i in to_remove or j in to_remove:
                continue

            r = corr_matrix.iat[i, j]
            if r > self.threshold:
                _, p_value = pearsonr(X_df.iloc[:, i], X_df.iloc[:, j])
                if p_value < self.alpha:
                    remove_idx = self._rng.choice([i, j])
                    to_remove.add(remove_idx)

        self.selected_features_mask_ = np.array([
            False if i in to_remove else True for i in range(n_features)
        ])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            transformed = X.loc[:, self.selected_features_mask_]
        else:
            transformed = X[:, self.selected_features_mask_]

        if self._output_format == 'pandas':
            feature_names = self.get_feature_names_out()
            if isinstance(transformed, pd.DataFrame):
                return transformed
            else:
                return pd.DataFrame(transformed, columns=feature_names, index=getattr(X, 'index', None))
        else:
            return transformed.values if isinstance(transformed, pd.DataFrame) else transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'input_features_', None)
            if input_features is None:
                raise ValueError("fit deve essere chiamato prima di get_feature_names_out oppure specifica input_features.")
        return np.array(input_features)[self.selected_features_mask_]


class TargetEtaSquared(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_value=0.1, mode="absolute"):
        """
        Inizializza il selettore basato su Eta Squared (η²) per la selezione delle feature in base alla loro 
        relazione con un target categoriale.

        :param threshold_value: Soglia di Eta Squared (η²) sopra la quale le feature vengono selezionate.
                                Se mode è 'absolute', il valore è una soglia fissa (es. 0.1).
                                Se mode è 'percentile', il valore è una percentuale del punteggio massimo di η².
        :param mode: Modalità per calcolare la soglia di selezione delle feature.
                     'absolute' per usare una soglia assoluta (es. 0.1),
                     'percentile' per usare un percentile (0-100) dei punteggi η².
        """
        # Controlla che il parametro mode sia valido
        if mode not in ["absolute", "percentile"]:
            raise ValueError(f"Il parametro mode deve essere 'absolute' o 'percentile'. Valore fornito: {mode}")
        
        # Controlla che il valore della soglia sia compreso tra 0 e 1
        if not (0 < threshold_value < 1):
            raise ValueError("threshold_value deve essere strettamente compreso tra 0 e 1. Valore fornito: {}".format(threshold_value))
        
        self.threshold_value = threshold_value
        self.mode = mode
        self._output_format = 'default'  # Impostazione predefinita per l'output: array NumPy

    def set_output(self, *, transform=None):
        """
        Imposta il formato dell'output del metodo transform.

        :param transform: 'default' (array NumPy) o 'pandas' (DataFrame). Se None, viene utilizzato il formato di default.
        :return: Restituisce l'oggetto stesso per un uso a catena del metodo.
        """
        # Verifica che il parametro di output sia valido
        if transform not in [None, 'default', 'pandas']:
            raise ValueError(f"transform deve essere 'default', 'pandas' o None. Valore fornito: {transform}")
        
        # Imposta il formato di output
        self._output_format = 'default' if transform is None else transform
        return self

    def eta_squared(self, categories, values):
        """
        Calcola eta squared (η²) tra una feature numerica e un target categoriale.

        :param categories: array o lista con il target categoriale.
        :param values: array o lista con la feature numerica.
        :return: Valore di eta squared (η²) che misura la forza della relazione tra feature e target.
        """
        # Converte in array numpy per garantire coerenza
        categories = np.array(categories)
        values = np.array(values)
        
        # Media complessiva della feature numerica
        overall_mean = np.mean(values)
        
        # Calcola le medie per ciascun gruppo nel target categoriale
        category_means = [np.mean(values[categories == cat]) for cat in np.unique(categories)]
        
        # Somma dei quadrati tra i gruppi (SS_between)
        ss_between = sum(len(values[categories == cat]) * (mean - overall_mean) ** 2 
                         for cat, mean in zip(np.unique(categories), category_means))
        
        # Somma totale dei quadrati (SS_total)
        ss_total = np.sum((values - overall_mean) ** 2)
        
        # Calcola eta squared (η²) come rapporto fra SS_between e SS_total
        eta_squared_value = ss_between / ss_total if ss_total > 0 else 0
        return eta_squared_value

    def fit(self, X, y=None):
        """
        Calcola Eta Squared per ogni feature e seleziona quelle che soddisfano la soglia impostata.

        :param X: DataFrame o array numpy contenente le features.
        :param y: Target categoriale (array o pandas Series) per il calcolo di Eta Squared.
        :return: Restituisce l'oggetto stesso per un uso a catena del metodo.
        """
        # Verifica che il target y sia stato fornito
        if y is None:
            raise ValueError("Il target 'y' deve essere fornito come input.")
            
        # Verifica se X è un DataFrame
        self._is_dataframe = isinstance(X, pd.DataFrame)
        
        # Se X è un DataFrame, convertilo in array numpy per l'elaborazione
        X_values = X.values if self._is_dataframe else X
        
        # Controllo su eventuali NaN nei dati
        if np.isnan(X_values).any():
            print("Errore: sono presenti valori NaN nei dati forniti a 'fit'. Rimuoverli o imputarli prima.")
            raise ValueError("Valori NaN trovati in X.")
        
        # Calcola eta squared per ogni feature
        eta_squared_scores = []
        for i in range(X_values.shape[1]):  # Per ogni feature in X
            eta_squared_scores.append(self.eta_squared(y, X_values[:, i]))  # Calcola η² per ogni colonna
        
        # Imposta la soglia per la selezione delle feature
        if self.mode == "percentile":
            # Usa il percentile se la modalità è 'percentile'
            threshold = np.percentile(eta_squared_scores, self.threshold_value * 100)
        elif self.mode == "absolute":
            # Usa il valore assoluto se la modalità è 'absolute'
            threshold = self.threshold_value
    
        self.threshold_ = threshold
        
        # Seleziona le feature che hanno un eta squared maggiore della soglia
        self.selected_features_ = [i for i, score in enumerate(eta_squared_scores) if score >= threshold]
        
        # Salva i nomi originali delle feature se l'input era un DataFrame
        if self._is_dataframe:
            self.input_features_ = X.columns
        else:
            self.input_features_ = [f"x{i}" for i in range(X.shape[1])]
        
        return self

    def transform(self, X):
        """
        Trasforma il dataset selezionando solo le feature che sono rimaste dopo la selezione.

        :param X: DataFrame o array numpy contenente le features.
        :return: Dati trasformati, ovvero solo le feature selezionate.
        """
        # Se X è un DataFrame, usa il metodo loc per selezionare le colonne
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        transformed = X_values[:, self.selected_features_]

        # Restituisce il formato di output richiesto
        if self._output_format == 'pandas':
            selected_names = self.get_feature_names_out(self.input_features_)
            return pd.DataFrame(transformed, columns=selected_names, index=getattr(X, 'index', None))
        return transformed

    def fit_transform(self, X, y=None):
        """
        Esegue fit e poi trasforma i dati in un solo passaggio.

        :param X: DataFrame o array numpy contenente le features.
        :param y: Target categoriale (array o pandas Series) per il calcolo di Eta Squared.
        :return: Le feature selezionate dopo il fitting e la trasformazione.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Restituisce i nomi delle feature selezionate.

        :param input_features: Lista/array dei nomi originali.
        :return: Nomi delle feature selezionate.
        """
        # Se non vengono forniti i nomi delle feature, usa quelli memorizzati durante il fitting
        if input_features is None:
            input_features = getattr(self, 'input_features_', None)
            if input_features is None:
                raise ValueError("fit deve essere chiamato prima di get_feature_names_out oppure specifica input_features.")
        # Restituisce i nomi delle feature selezionate tramite la maschera
        return np.array(input_features)[self.selected_features_]

    
class TargetPearson(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_value=0.1, p_value_threshold=0.05, mode="absolute"):
        """
        Inizializza il selettore basato sulla correlazione di Pearson tra features e target.
        
        :param threshold_value: Soglia per la selezione delle feature (può essere un valore assoluto o un percentile).
        :param p_value_threshold: Soglia per il p-value per considerare la correlazione significativa.
        :param mode: Modalità per determinare la soglia di correlazione.
                     'absolute' per una soglia assoluta, 'percentile' per un percentile della correlazione di Pearson.
        """
        # Verifica che il parametro 'mode' sia valido
        if mode not in ["percentile", "absolute"]:
            raise ValueError(f"Il parametro mode deve essere 'percentile' o 'absolute'. Valore fornito: {mode}")
        
        # Verifica che il parametro 'threshold_value' sia compreso tra 0 e 1
        if not (0 <= threshold_value <= 1):
            raise ValueError("threshold_value deve essere un numero compreso tra 0 e 1. Valore fornito: {}".format(threshold_value))
        
        # Verifica che il parametro 'p_value_threshold' sia compreso tra 0 e 1
        if not (0 <= p_value_threshold <= 1):
            raise ValueError("p_value_threshold deve essere un numero compreso tra 0 e 1. Valore fornito: {}".format(p_value_threshold))

        self.threshold_value = threshold_value
        self.p_value_threshold = p_value_threshold
        self.mode = mode

    def fit(self, X, y=None):
        """
        Esegue la selezione delle feature in base alla correlazione di Pearson tra le features e il target.
        
        :param X: DataFrame o array numpy contenente le feature.
        :param y: Target numerico continuo (array o pandas Series).
        :return: Seleziona le feature in base alla correlazione con il target e al p-value.
        """
        # Verifica che il target y sia stato fornito
        if y is None:
            raise ValueError("Il target 'y' deve essere fornito come input.")
            
        # Verifica che il target 'y' sia numerico
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Il target 'y' deve essere numerico.")
        
        # Verifica se X è un DataFrame
        self._is_dataframe = isinstance(X, pd.DataFrame)
        
        # Se X è un DataFrame, convertilo in array numpy per l'elaborazione
        X_values = X.values if self._is_dataframe else X
        
        # Controllo su eventuali NaN nei dati
        if np.isnan(X_values).any():
            print("Errore: sono presenti valori NaN nei dati forniti a 'fit'. Rimuoverli o imputarli prima.")
            raise ValueError("Valori NaN trovati in X.")
        
        # Calcola la correlazione di Pearson e il p-value tra ogni feature e il target
        correlations = []
        p_values = []
        for i in range(X_values.shape[1]):  # Itera su ogni feature di X
            corr, p_val = pearsonr(X_values[:, i], y)  # Calcola la correlazione e il p-value
            correlations.append(abs(corr))  # Usa il valore assoluto della correlazione per ignorare la direzione
            p_values.append(p_val)  # Salva il p-value per ciascuna feature

        # Imposta la soglia per la selezione delle feature in base al 'mode' scelto
        if self.mode == "percentile":
            # Calcola il percentile della correlazione
            threshold = np.percentile(correlations, self.threshold_value * 100)
        elif self.mode == "absolute":
            # Usa una soglia assoluta
            threshold = self.threshold_value

        self.threshold_ = threshold
        
        # Seleziona le feature che hanno una correlazione maggiore della soglia e un p-value significativo
        self.selected_features_ = [
            i for i, (score, p_val) in enumerate(zip(correlations, p_values)) 
            if score >= threshold and p_val <= self.p_value_threshold
        ]
        
        # Salva i nomi originali delle feature se l'input è un DataFrame
        if self._is_dataframe:
            self.input_features_ = X.columns
        else:
            self.input_features_ = [f"x{i}" for i in range(X.shape[1])]
        
        return self

    def transform(self, X):
        """
        Restituisce le feature selezionate in base alla correlazione e al p-value.
        
        :param X: DataFrame o array numpy contenente le feature.
        :return: Le feature selezionate in base alla correlazione con il target.
        """
        # Converte X in un array NumPy se è un DataFrame
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        # Seleziona solo le colonne delle feature che sono state selezionate
        transformed = X_values[:, self.selected_features_]
        
        # Restituisce il formato di output richiesto (pandas o NumPy)
        if self._is_dataframe:
            selected_names = self.get_feature_names_out(self.input_features_)
            return pd.DataFrame(transformed, columns=selected_names, index=getattr(X, 'index', None))
        return transformed
    
    def fit_transform(self, X, y):
        """
        Applica il metodo fit e poi trasforma i dati in un solo passaggio.
        
        :param X: DataFrame o array numpy contenente le feature.
        :param y: Target numerico continuo.
        :return: Le feature selezionate in base alla correlazione con il target.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Restituisce i nomi delle feature selezionate.
        
        :param input_features: Lista/array dei nomi originali delle feature.
        :return: Nomi delle feature selezionate.
        """
        # Se non vengono forniti i nomi delle feature, usa quelli memorizzati durante il fitting
        if input_features is None:
            input_features = getattr(self, 'input_features_', None)
            if input_features is None:
                raise ValueError("fit deve essere chiamato prima di get_feature_names_out oppure specifica input_features.")
        
        # Restituisce i nomi delle feature selezionate
        return np.array(input_features)[self.selected_features_]