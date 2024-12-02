
# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

import time
import psutil
from math import ceil

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve
)
import shap

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer



class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42, shuffle=True
        )

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(),
            "tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier()
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, "pred_test": None} for nombre in self.modelos}

    def get_X_y(self):
        return self.X, self.y
    
    def get_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_modelos(self):
        return self.resultados
    
    def ajustar_modelo(self, modelo_nombre, param_grid=None):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo
        parametros_default = {
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 6, 8, 20, 12, 16],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "gradient_boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})

        if modelo_nombre == "logistic_regression":
            modelo_logistica = LogisticRegression(random_state=42)
            modelo_logistica.fit(self.X_train, self.y_train)
            self.resultados[modelo_nombre]["pred_train"] = modelo_logistica.predict(self.X_train)
            self.resultados[modelo_nombre]["pred_test"] = modelo_logistica.predict(self.X_test)
            self.resultados[modelo_nombre]["mejor_modelo"] = modelo_logistica

        else:
            # Ajuste del modelo
            grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            print(f"El mejor modelo es {grid_search.best_estimator_}")
            self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_
            self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
            self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)

        
    def calcular_metricas(self, modelo_nombre):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC, Kappa,
        tiempo de computación y núcleos utilizados.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        pred_train = self.resultados[modelo_nombre]["pred_train"]
        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_train is None or pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        # Registrar tiempo de ejecución
        start_time = time.time()
        if hasattr(modelo, "predict_proba"):
            prob_train = modelo.predict_proba(self.X_train)[:, 1]
            prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = prob_test = None
        elapsed_time = time.time() - start_time

        # Registrar núcleos utilizados
        num_nucleos = getattr(modelo, "n_jobs", psutil.cpu_count(logical=True))

        # Métricas para conjunto de entrenamiento
        metricas_train = {
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, prob_train) if prob_train is not None else None,
        }

        # Métricas para conjunto de prueba
        metricas_test = {
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, prob_test) if prob_test is not None else None,
        }

        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test}).T

    def plot_matriz_confusion(self, modelo_nombre):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, pred_test)
        plt.figure(figsize=(3, 3))
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo_nombre})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()

    def importancia_predictores(self, modelo_nombre):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(modelo, "feature_importances_"):
            importancia = modelo.feature_importances_
        elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
            importancia = modelo.coef_[0]
        else:
            print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
            return
        
        # Crear DataFrame y graficar
        importancia_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 7))
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.show()

    def curva_roc(self, modelo_nombre):

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        y_pred_test_prob = modelo.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_test_prob)
        plt.figure(figsize=(7,5))
        sns.lineplot(x=[0,1], y=[0,1], color="grey")
        sns.lineplot(x=fpr, y=tpr, color="blue")
        plt.xlabel("Ratios Falsos Positivos : 1-Especificidad")
        plt.ylabel("Ratios Verdaderos Positivos : Recall")
        plt.title("Curva ROC")

    def plot_shap_summary(self, modelo_nombre, plot_size=(9,5)):
        """
        Genera un SHAP summary plot para el modelo seleccionado.
        Maneja correctamente modelos de clasificación con múltiples clases.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de generar el SHAP plot.")

        # Usar TreeExplainer para modelos basados en árboles
        if modelo_nombre in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(self.X_test)

            # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
            if isinstance(shap_values, list):
                # Para modelos binarios, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Para Decision Trees, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[:, :, 1]
        else:
            # Usar el explicador genérico para otros modelos
            explainer = shap.Explainer(modelo, self.X_test, check_additivity=False)
            shap_values = explainer(self.X_test).values

        # Generar el summary plot estándar
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns, plot_size=plot_size)

    def plot_all_matriz_confusion(self):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """

        lista_modelos = list(self.resultados.items())
        num_modelos = len(lista_modelos)

        if num_modelos == 0:
            print("No hay modelos disponibles para graficar.")
            return
        
        filas = ceil(num_modelos/2)
        columnas = 2

        fig, axes = plt.subplots(filas, columnas, figsize=(10,4))
        axes = axes.flat


        for i, mod in enumerate(lista_modelos):
            modelo = mod[0]
            valores_modelo = mod[1]
            pred_test = valores_modelo["pred_test"]

            if pred_test is None:
                pass

            # Matriz de confusión
            matriz_conf = confusion_matrix(self.y_test, pred_test)
            sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues', ax=axes[i])
            axes[i].set_title(f"Matriz de Confusión ({modelo})")
            axes[i].set_xlabel("Predicción")
            axes[i].set_ylabel("Valor Real")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    def curva_roc(self, modelo_nombre):

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        y_pred_test_prob = modelo.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_test_prob)
        plt.figure(figsize=(7,5))
        sns.lineplot(x=[0,1], y=[0,1], color="grey")
        sns.lineplot(x=fpr, y=tpr, color="blue")
        plt.xlabel("Ratios Falsos Positivos : 1-Especificidad")
        plt.ylabel("Ratios Verdaderos Positivos : Recall")
        plt.title("Curva ROC")


    def plot_curvas_roc_train_test(self):
        """
        Plotea las curvas ROC para todos los modelos en los datos de entrenamiento y prueba en subplots.
        """

        lista_modelos = list(self.resultados.items())
        num_modelos = len(lista_modelos)

        if num_modelos == 0:
            print("No hay modelos disponibles para graficar.")
            return

        # Crear los subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for idx, (dataset, X, y, title, ax) in enumerate([
            ("train", self.X_train, self.y_train, "Curvas ROC (Train)", axes[0]),
            ("test", self.X_test, self.y_test, "Curvas ROC (Test)", axes[1]),
        ]):
            for mod in lista_modelos:
                modelo = mod[0]
                valores_modelo = mod[1]
                mejor_modelo = valores_modelo.get("mejor_modelo", None)

                if mejor_modelo is None:
                    print(f"Modelo {modelo} no tiene un modelo ajustado para {dataset}.")
                    continue

                # Predicciones de probabilidad
                y_pred_prob = mejor_modelo.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_pred_prob)

                # Graficar la curva ROC
                sns.lineplot(x=fpr, y=tpr, label=f"{modelo}", ax=ax)

            # Línea diagonal base
            sns.lineplot(x=[0, 1], y=[0, 1], color="grey", linestyle="--", label="Referencia (AUC: 0.50)", ax=ax)

            # Configuración del subplot
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.grid(True)

        # Ajustar diseño general
        plt.tight_layout()
        plt.show()

# Función para asignar colores
def color_filas_con_borde(row):
    styles = []

    for col in row.index:
        # Condición para pintar la celda de 'auc' en rojo si el valor es menor a 0.6
        if col == "kappa" and row["kappa"] < 0.6:
            styles.append("background-color: #ff9999; color: black; border-bottom: 1px solid #000000;")  # Rojo con borde fino
        elif row["modelo"] == "decision tree":
            styles.append("background-color: #e6b3e0; color: black; border-bottom: 1px solid #000000;")
        elif row["modelo"] == "random forest":
            styles.append("background-color: #c2f0c2; color: black; border-bottom: 1px solid #000000;")
        elif row["modelo"] == "gradient boosting":
            styles.append("background-color: #ffd9b3; color: black; border-bottom: 1px solid #000000;")
        elif row["modelo"] == "xgboost":
            styles.append("background-color: #f7b3c2; color: black; border-bottom: 1px solid #000000;")
        elif row["modelo"] == "regresion logistica":
            styles.append("background-color: #b3d1ff; color: black; border-bottom: 1px solid #000000;")
        else:
            styles.append("color: black; border-bottom: 1px solid #000000;")  # Texto negro con borde fino

    return styles

