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

#%% Libreries

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr


#%% Feature selection methods

# ==============================================================
# Variance feature selection (continuous numerical features)
# ==============================================================
class FeaturesVariance(BaseEstimator, TransformerMixin):
    """
    Feature selector that removes low-variance features from a dataset.

    This transformer selects features based on their variance, with two possible
    threshold modes:
    1. 'percentile': select features above a given percentile of variances.
    2. 'max_percentage': select features above a percentage of the maximum variance.

    Parameters
    ----------
    threshold_value : float, default=90
        Threshold for feature selection. Interpreted according to `mode`.
        Must be between 0 and 100.
    mode : {'percentile', 'max_percentage'}, default='percentile'
        Method to calculate the variance threshold:
        - 'percentile': select features with variance above the given percentile.
        - 'max_percentage': select features with variance above a percentage of the maximum variance.

    Attributes
    ----------
    threshold_ : float
        Calculated variance threshold after fitting.
    selected_features_ : ndarray of int
        Indices of the selected features.
    input_features_ : ndarray of str
        Names of the input features (from DataFrame columns or generated as x0, x1, ...).
    _output_format : str
        Output format, either 'default' (NumPy array) or 'pandas' (DataFrame).
    _is_dataframe : bool
        Whether the input X during fit was a DataFrame.
    """

    def __init__(self, threshold_value=90, mode="percentile"):
        if mode not in ["percentile", "max_percentage"]:
            raise ValueError(f"mode must be 'percentile' or 'max_percentage'. Got: {mode}")

        if not (0 < threshold_value < 100):
            raise ValueError(
                f"threshold_value must be strictly between 0 and 100. Got: {threshold_value}"
            )

        self.threshold_value = threshold_value
        self.mode = mode
        self._output_format = 'default'

    def set_output(self, *, transform=None):
        """
        Set the output format for the transform method.

        Parameters
        ----------
        transform : {'default', 'pandas', None}, optional
            - 'default': return a NumPy array (default).
            - 'pandas': return a DataFrame with column names.
            - None: keep the current format.

        Returns
        -------
        self : object
            Returns the transformer itself.
        """
        if transform not in [None, 'default', 'pandas']:
            raise ValueError(f"transform must be 'default', 'pandas', or None. Got: {transform}")
        self._output_format = 'default' if transform is None else transform
        return self

    def fit(self, X, y=None):
        """
        Compute feature variances and determine which features to keep.

        Parameters
        ----------
        X : {array-like, DataFrame} of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Check if input is a DataFrame
        self._is_dataframe = isinstance(X, pd.DataFrame)

        # Convert to NumPy array for calculations
        X_values = X.values if self._is_dataframe else np.asarray(X)

        # Raise error if NaNs are present
        if np.isnan(X_values).any():
            raise ValueError("NaN values detected in X. Remove or impute them before fitting.")

        # Scale features to [0, 1] range
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_values)

        # Compute variance for each feature
        variances = np.var(X_scaled, axis=0)

        # Determine threshold based on selected mode
        if self.mode == "percentile":
            threshold = np.percentile(variances, self.threshold_value)
        else:  # max_percentage
            threshold = (self.threshold_value / 100) * np.max(variances)

        self.threshold_ = threshold

        # Store indices of features above threshold
        self.selected_features_ = np.where(variances >= threshold)[0]

        # Store feature names
        if self._is_dataframe:
            self.input_features_ = np.array(X.columns)
        else:
            self.input_features_ = np.array([f"x{i}" for i in range(X.shape[1])])

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : {array-like, DataFrame} of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_reduced : {ndarray, DataFrame}
            Input data with only selected features.
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        transformed = X_values[:, self.selected_features_]

        # Return as DataFrame if requested
        if self._output_format == 'pandas':
            selected_names = self.get_feature_names_out()
            return pd.DataFrame(transformed, columns=selected_names, index=getattr(X, 'index', None))

        # Otherwise return NumPy array
        return transformed

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the selected features.

        Parameters
        ----------
        input_features : array-like of str, optional
            Original feature names. If None, uses names stored during fit.

        Returns
        -------
        selected_feature_names : ndarray of str
            Names of the selected features.

        Raises
        ------
        AttributeError
            If transformer is called before fitting and no input_features are provided.
        """
        if input_features is None:
            input_features = getattr(self, 'input_features_', None)
            if input_features is None:
                raise AttributeError("Transformer has not been fitted yet.")
        return np.array(input_features)[self.selected_features_]


# ==============================================================
# Pearson correlation feature selection (continuous numerical features)
# ==============================================================
class FeaturesPearson(BaseEstimator, TransformerMixin):
    """
    Feature selector based on Pearson correlation.

    Removes highly correlated features among numerical continuous columns.
    If two features are highly correlated and the correlation is statistically
    significant, the feature with lower variance is removed.

    Parameters
    ----------
    threshold : float, default=0.9
        Correlation threshold above which one of the two features is removed.
    alpha : float, default=0.05
        Significance level for the p-value of the correlation.
    random_state : int or None, default=None
        Seed for reproducibility (used if tie-breaking randomly; normally deterministic).

    Attributes
    ----------
    selected_features_mask_ : ndarray of bool
        Boolean mask indicating which features are selected.
    input_features_ : ndarray of str
        Names of the input features.
    _output_format : str
        Output format, either 'default' (NumPy array) or 'pandas' (DataFrame).
    """

    def __init__(self, threshold=0.9, alpha=0.05, random_state=None):
        self.threshold = threshold
        self.alpha = alpha
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._output_format = 'default'

    def set_output(self, *, transform=None):
        """Set the output format for transform: 'default' (NumPy) or 'pandas' DataFrame."""
        if transform not in [None, 'default', 'pandas']:
            raise ValueError(f"transform must be 'default', 'pandas' or None. Got: {transform}")
        self._output_format = 'default' if transform is None else transform
        return self

    def fit(self, X, y=None):
        """
        Fit the feature selector on the input data.

        Parameters
        ----------
        X : {DataFrame, ndarray} of shape (n_samples, n_features)
            Input data to analyze.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
        """
        # Convert to DataFrame for convenience
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self._is_dataframe = True
        else:
            X_df = pd.DataFrame(X)
            self._is_dataframe = False

        # Check that all columns are numeric
        if not np.all([np.issubdtype(dtype, np.number) for dtype in X_df.dtypes]):
            raise TypeError("All columns must be numeric continuous. Non-numeric columns found.")

        self.input_features_ = np.array(X_df.columns)

        # Min-Max scale features (mandatory)
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

        # Compute variance for deterministic removal
        variances = X_scaled.var()

        # Initialize mask for selected features
        n_features = X_df.shape[1]
        selected_mask = np.ones(n_features, dtype=bool)

        # Compute correlation matrix (absolute values)
        corr_matrix = X_df.corr().abs()
        upper_tri_indices = np.triu_indices(n_features, k=1)

        # Loop over feature pairs
        for i, j in zip(*upper_tri_indices):
            if not selected_mask[i] or not selected_mask[j]:
                # Skip if either feature is already removed
                continue

            r = corr_matrix.iat[i, j]

            if r > self.threshold:
                _, p_value = pearsonr(X_df.iloc[:, i], X_df.iloc[:, j])
                if p_value < self.alpha:
                    # Remove feature with lower variance
                    if variances[i] < variances[j]:
                        selected_mask[i] = False
                    elif variances[j] < variances[i]:
                        selected_mask[j] = False
                    else:
                        # If equal, remove randomly
                        remove_idx = self._rng.choice([i, j])
                        selected_mask[remove_idx] = False

        self.selected_features_mask_ = selected_mask
        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : {DataFrame, ndarray} of shape (n_samples, n_features)

        Returns
        -------
        X_reduced : {ndarray, DataFrame}
        """
        if self._is_dataframe:
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
                raise ValueError("fit must be called before get_feature_names_out or provide input_features.")
        return np.array(input_features)[self.selected_features_mask_]


# ==============================================================
# Eta**2 feature selection (continuous numerical features - categorical target)
# ==============================================================
class TargetEtaSquared(BaseEstimator, TransformerMixin):
    """
    Feature selector based on Eta Squared (η²) for categorical targets.

    This selector computes the strength of the relationship between each numeric feature
    and a categorical target using Eta Squared. Features with η² above a given threshold
    are selected.

    Parameters
    ----------
    threshold_value : float, default=0.1
        Threshold for selecting features. Must be between 0 and 1.
        - If mode='absolute', features with η² >= threshold_value are selected.
        - If mode='percentile', features with η² above the given percentile of all scores
          are selected.
    mode : {'absolute', 'percentile'}, default='absolute'
        Mode for thresholding. 'absolute' uses a fixed η² threshold, 'percentile' uses
        a percentile of the η² scores.

    Attributes
    ----------
    selected_features_ : list of int
        Indices of features selected after fitting.
    threshold_ : float
        Threshold value used to select features after fitting.
    input_features_ : list of str
        Original feature names (if DataFrame provided).
    _output_format : str
        Output format of transform: 'default' (NumPy) or 'pandas' (DataFrame).
    """

    def __init__(self, threshold_value=0.1, mode="absolute"):
        if mode not in ["absolute", "percentile"]:
            raise ValueError(f"Mode must be 'absolute' or 'percentile'. Got: {mode}")
        if not (0 < threshold_value < 1):
            raise ValueError(f"threshold_value must be between 0 and 1. Got: {threshold_value}")

        self.threshold_value = threshold_value
        self.mode = mode
        self._output_format = 'default'

    def set_output(self, *, transform=None):
        """
        Set output format for transform.

        Parameters
        ----------
        transform : {'default', 'pandas'} or None
            Output format. 'default' returns a NumPy array, 'pandas' returns a DataFrame.
            If None, defaults to 'default'.

        Returns
        -------
        self
        """
        if transform not in [None, 'default', 'pandas']:
            raise ValueError(f"transform must be 'default', 'pandas' or None. Got: {transform}")
        self._output_format = 'default' if transform is None else transform
        return self

    def eta_squared(self, categories, values):
        """
        Compute eta squared (η²) for a single feature and categorical target.

        Parameters
        ----------
        categories : array-like
            Categorical target labels.
        values : array-like
            Numeric feature values.

        Returns
        -------
        float
            Eta squared (η²) value, between 0 and 1.
        """
        categories = np.array(categories)
        values = np.array(values)
        overall_mean = np.mean(values)

        # Group by category using pandas for speed
        df = pd.DataFrame({'target': categories, 'feature': values})
        grouped = df.groupby('target')['feature']
        group_counts = grouped.count()
        group_means = grouped.mean()

        # Sum of squares between groups
        ss_between = np.sum(group_counts * (group_means - overall_mean) ** 2)

        # Total sum of squares
        ss_total = np.sum((values - overall_mean) ** 2)

        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        return eta_sq

    def fit(self, X, y=None):
        """
        Fit the selector by computing η² for each feature and selecting those above the threshold.

        Parameters
        ----------
        X : {DataFrame, ndarray} of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            Categorical target.

        Returns
        -------
        self
        """
        if y is None:
            raise ValueError("Target 'y' must be provided.")

        # Convert to DataFrame for convenience
        self._is_dataframe = isinstance(X, pd.DataFrame)
        X_values = X.values if self._is_dataframe else X

        if np.isnan(X_values).any():
            raise ValueError("NaN values found in X. Please remove or impute them before fitting.")

        eta_scores = []
        for i in range(X_values.shape[1]):
            eta_scores.append(self.eta_squared(y, X_values[:, i]))

        # Compute threshold based on mode
        if self.mode == "percentile":
            threshold = np.percentile(eta_scores, self.threshold_value * 100)
        else:
            threshold = self.threshold_value

        self.threshold_ = threshold
        self.selected_features_ = [i for i, score in enumerate(eta_scores) if score >= threshold]

        # Save original feature names if DataFrame
        self.input_features_ = X.columns if self._is_dataframe else [f"x{i}" for i in range(X_values.shape[1])]

        return self

    def transform(self, X):
        """
        Transform X to keep only selected features.

        Parameters
        ----------
        X : {DataFrame, ndarray}

        Returns
        -------
        X_reduced : {DataFrame, ndarray}
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        transformed = X_values[:, self.selected_features_]

        if self._output_format == 'pandas':
            feature_names = self.get_feature_names_out()
            return pd.DataFrame(transformed, columns=feature_names, index=getattr(X, 'index', None))
        return transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return names of selected features."""
        if input_features is None:
            input_features = getattr(self, 'input_features_', None)
            if input_features is None:
                raise ValueError("fit must be called before get_feature_names_out or provide input_features.")
        return np.array(input_features)[self.selected_features_]


# ==============================================================
# Eta**2 feature selection (continuous numerical features - categorical target)
# ==============================================================
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