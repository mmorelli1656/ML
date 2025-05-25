# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:35:23 2025

@author: mik16
"""

#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


#%%

class EvaluationMetrics:
    def __init__(self, df_pred, df_pred_proba=None, task="binaryclass"):
        """
        Classe per calcolare metriche di valutazione per classificazione binaria, multiclasse e regressione.
        
        :param df_pred: DataFrame contenente le predizioni (prima colonna = true labels, successive = predizioni)
        :param df_pred_proba: DataFrame contenente le probabilità previste (necessario per ROC-AUC)
        :param task: "binary" per classificazione binaria, "multiclass" per multiclasse, "regression" per regressione
        """
        self.df_pred = df_pred
        self.df_pred_proba = df_pred_proba
        self.task = task
        self.y_true = df_pred.iloc[:, 0].values  # Label reali
        
    def _binary_classification_metrics(self):
        """Calcola metriche per la classificazione binaria."""
        metrics_dict = {"Accuracy": [], "Precision": [], "Recall": [], "Specificity": [], "F1-score": []}

        for col in self.df_pred.columns[1:]:
            y_pred = self.df_pred[col].values

            metrics_dict["Accuracy"].append(accuracy_score(self.y_true, y_pred))
            metrics_dict["Precision"].append(precision_score(self.y_true, y_pred, zero_division=0))
            metrics_dict["Recall"].append(recall_score(self.y_true, y_pred))
            metrics_dict["F1-score"].append(f1_score(self.y_true, y_pred))

            # Specificità (TN / (TN + FP))
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics_dict["Specificity"].append(specificity)

            # AUC: calcoliamo solo se df_pred_proba è passato
            if self.df_pred_proba is not None:
                metrics_dict["AUC"] = []
                for col in self.df_pred.columns[1:]:
                    metrics_dict["AUC"].append(roc_auc_score(self.y_true, self.df_pred_proba[col]))

        return metrics_dict
    
    def _multiclass_classification_metrics(self):
        """Calcola metriche per la classificazione multiclasse."""
        metrics_dict = {"Accuracy": [], "Macro Precision": [], "Macro Recall": [], "Macro F1-score": []}
        
        for col in self.df_pred.columns[1:]:
            y_pred = self.df_pred[col].values

            metrics_dict["Accuracy"].append(accuracy_score(self.y_true, y_pred))
            metrics_dict["Macro Precision"].append(precision_score(self.y_true, y_pred, average="macro", zero_division=0))
            metrics_dict["Macro Recall"].append(recall_score(self.y_true, y_pred, average="macro"))
            metrics_dict["Macro F1-score"].append(f1_score(self.y_true, y_pred, average="macro"))

        return metrics_dict
    
    def _regression_metrics(self):
        """Calcola metriche per la regressione."""
        metrics_dict = {"MAE": [], "MAPE": [], "RMSE": [], "R2": []}
        
        for col in self.df_pred.columns[1:]:
            y_pred = self.df_pred[col].values
            
            metrics_dict["MAE"].append(mean_absolute_error(self.y_true, y_pred))
            metrics_dict["MAPE"].append(mean_absolute_percentage_error(self.y_true, y_pred))
            metrics_dict["RMSE"].append(np.sqrt(mean_squared_error(self.y_true, y_pred)))
            metrics_dict["R2"].append(r2_score(self.y_true, y_pred))

        return metrics_dict

    def compute_metrics(self):
        """Calcola le metriche in base alla tipologia di task e stampa i risultati."""
        if self.task == "binaryclass":
            metrics_dict = self._binary_classification_metrics()
        elif self.task == "multiclass":
            metrics_dict = self._multiclass_classification_metrics()
        elif self.task == "regression":
            metrics_dict = self._regression_metrics()
        else:
            raise ValueError("Task non riconosciuto! Usa 'binaryclass', 'multiclass' o 'regression'.")

        # Creazione del DataFrame con le metriche
        df_metrics = pd.DataFrame(metrics_dict)

        # Stampa dei risultati in forma compatta
        print("\n--- Risultati delle metriche ---")
        for metric, values in metrics_dict.items():
            median_value = np.median(values)
            iqr_value = np.percentile(values, 75) - np.percentile(values, 25)
            print(f"{metric}: Mediana = {median_value:.4f}, IQR = {iqr_value:.4f}")

        return df_metrics

    def plot_confusion_matrix(self, perc='row', classes=None, save=False):
        """Genera la matrice di confusione per classificazione binaria e multiclasse."""
        if self.task not in ["binaryclass", "multiclass"]:
            raise ValueError("La matrice di confusione è disponibile solo per la classificazione.")
        
        def cm(df_pred):
            # Estrarre le etichette reali
            y_true = df_pred.iloc[:, 0]
            
            # Creare una lista per raccogliere le matrici di confusione per ogni colonna di predizione
            confusion_matrices = []
            
            for col in df_pred.columns[1:]:
                y_pred = df_pred[col]
                cm = confusion_matrix(y_true, y_pred)
                confusion_matrices.append(cm)
            
            # Convertire la lista in un array numpy
            confusion_matrices = np.array(confusion_matrices)
            
            # Calcolare la mediana e l'IQR lungo la prima dimensione (ripetizioni CV)
            cm_median = np.median(confusion_matrices, axis=0)
            cm_q1 = np.percentile(confusion_matrices, 25, axis=0)
            cm_q3 = np.percentile(confusion_matrices, 75, axis=0)
            cm_iqr = cm_q3 - cm_q1
            
            return cm_median, cm_iqr
        
        def plot_cm(cm_median, cm_iqr, perc='row', classes=None):
            
            if classes is None:
                # Determinare il numero di classi
                classes = np.unique(self.y_true)
               
            if perc == "row":
                cm_median_perc = cm_median / cm_median.sum(axis=1, keepdims=True) * 100  # Percentuale per riga
                cm_iqr_perc = cm_iqr / cm_median.sum(axis=1, keepdims=True) * 100  # Percentuale per riga
            elif perc == "total":
                cm_total = cm_median.sum()
                cm_median_perc = cm_median / cm_total * 100  # Percentuale totale
                cm_iqr_perc = cm_iqr / cm_total * 100
            else:
                raise ValueError("La percentuale può essere calcolata o sul numero di elementi sulla riga (row) o sul numero di elementi totale (total).")
            
            # Creare la matrice con mediana ± IQR e percentuale
            cm_display = np.char.array(cm_median.astype(str))
            for i in range(cm_median.shape[0]):
                for j in range(cm_median.shape[1]):
                    cm_display[i, j] += f' ± {cm_iqr[i, j]:.2f}\n{cm_median_perc[i, j]:.2f} ± {cm_iqr_perc[i, j]:.2f} %'
            
            # Plot della matrice con mediana ± IQR e percentuale
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_median, annot=cm_display, fmt='', cmap='YlGnBu',
                        xticklabels=classes, yticklabels=classes, annot_kws={'size': 14})
            plt.xlabel('Predicted', fontsize=14)
            plt.ylabel('Actual', fontsize=14)
            plt.title('Confusion Matrix (Median ± IQR)', fontsize=16)
            
            # Salvataggio
            if save:
                filename = input("Inserisci il nome del file per salvare la figura (es: 'conf_matrix.png'): ")
                plt.savefig(f"03_Images/CM_{filename}", bbox_inches='tight')
                print(f"Immagine salvata come {filename}")
            
            plt.show()
            plt.close()

        # Calcolare la matrice di confusione
        cm_median, cm_iqr = cm(self.df_pred)
        
        # Generare il plot
        plot_cm(cm_median, cm_iqr, perc, classes)
        
    def plot_roc_curve(self, save=False):
        """
        Plot delle curve ROC per ciascun set di predizioni probabilistiche
        contenute in self.df_pred_proba. Si assume che la prima colonna
        sia 'true_labels' e le successive siano le predizioni.
        """
        if not hasattr(self, "df_pred_proba") or self.df_pred_proba is None:
            raise AttributeError("df_pred_proba non è stato trovato o è None.")

        if not hasattr(self, "task") or self.task != "binaryclass":
            raise ValueError("Il metodo plot_roc_curve è disponibile solo per problemi di classificazione binaria.")

        plt.figure(figsize=(8, 6))
    
        # Plot curva random
        plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
        
        auc_list = []
        
        # ROC curves rosse senza label
        for col in self.df_pred_proba.columns[1:]:
            y_scores = self.df_pred_proba[col]
            fpr, tpr, _ = roc_curve(self.y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='red', alpha=0.4)
        
            auc_list.append(roc_auc)
        
        # Calcola statistica AUC
        auc_median = np.median(auc_list)
        auc_iqr = np.percentile(auc_list, 75) - np.percentile(auc_list, 25)
        
        # Linee fittizie per la legenda
        plt.plot([], [], color='red', label='ROC')
        plt.plot([], [], color='white', label=f'Median AUC = {auc_median:.2f} ± {auc_iqr:.2f}')        
        
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Salvataggio
        if save:
            filename = input("Inserisci il nome del file per salvare la figura (es: 'conf_matrix.png'): ")
            plt.savefig(f"03_Images/ROC_{filename}", bbox_inches='tight')
            print(f"Immagine salvata come {filename}")
            
        plt.show()
        plt.close()

    def plot_metrics_boxplot(self, df_metrics, save=False):
        """
        Plot separati (in griglia 2x2) per ogni metrica di regressione.
        Per classificazione, mantiene il boxplot unico.
        """
        if df_metrics is None or df_metrics.empty:
            raise ValueError("Il DataFrame df_metrics non può essere vuoto.")
    
        if self.task == "regression":
            metrics = ['MAE', 'MAPE', 'RMSE', 'R2']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                sns.boxplot(y=df_metrics[metric], ax=axes[i], color=colors[i],
                            showfliers=True, width=0.3)
                axes[i].set_title(metric, fontsize=16)
                axes[i].set_ylabel('Values', fontsize=12)
    
            plt.suptitle('Metrics Boxplots', fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 1])
    
            # Salvataggio
            if save:
                filename = input("Inserisci il nome del file per salvare la figura (es: 'boxplot_metrics.png'): ")
                plt.savefig(f"03_Images/BP_{filename}", bbox_inches='tight')
                print(f"Immagine salvata come {filename}")
    
            plt.show()
            plt.close()
    
        else:
            # Per classificazione: boxplot unico
            df_melted = df_metrics.melt(var_name='Metric', value_name='Value')
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Metric', y='Value', data=df_melted,
                        hue='Metric', palette='Set2', legend=False)
            plt.xlabel('Metrics', fontsize=14)
            plt.ylabel('Values', fontsize=14)
            plt.title('Metrics Boxplot', fontsize=16)
            plt.xticks(rotation=45, fontsize=12)
            plt.tight_layout()
    
            if save:
                filename = input("Inserisci il nome del file per salvare la figura (es: 'boxplot_metrics.png'): ")
                plt.savefig(f"03_Images/BP_{filename}", bbox_inches='tight')
                print(f"Immagine salvata come {filename}")
    
            plt.show()
            plt.close()
