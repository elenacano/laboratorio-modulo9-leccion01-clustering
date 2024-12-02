from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def estandarizacion(df, col_num, modelos_estand):

    dic_scalers={}

    for estand in modelos_estand:

        if estand.lower() == "robust":
            escalador = RobustScaler()
            colums_escaladas = [f"{elem}_robust" for elem in col_num]
            
        elif estand.lower() == "standar":
            escalador = StandardScaler()
            colums_escaladas = [f"{elem}_standar" for elem in col_num]
            
        elif estand.lower() == "minmax":
            escalador = MinMaxScaler()
            colums_escaladas = [f"{elem}_minmax" for elem in col_num]

        elif estand.lower() == "normalizer":
            escalador = Normalizer()
            colums_escaladas = [f"{elem}_normalizer" for elem in col_num]
        else:
            print("Escalador erroneo, por favor introduzca una de las siguientes opciones:")
            print(" - robust\n - standar\n - minmax\n - normalizer")
            break
            
        datos_transf = escalador.fit_transform(df[col_num])
        df[colums_escaladas] = datos_transf
        dic_scalers[estand.lower()]=escalador

    return df, dic_scalers


def visualizacion_estandarizacion(data, columnas, num_columnas, figsize=(15,10)):

    num_filas = math.ceil(len(columnas) / num_columnas)

    fig, axes = plt.subplots(nrows=num_filas, ncols=num_columnas, figsize=figsize)
    axes = axes.flat

    for i, col in enumerate(columnas):
        sns.boxplot(x=col, data=data, ax=axes[i])
        axes[i].set_title(f"Boxplot de {col}")

    for j in range(len(columnas), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualizacion_boxplot_hisplot_estand(df, columnas, modelos_estand, figsize = (15,25)):
    lista_final = []

    for col in columnas:
        lista_final.append(col)
        for modelo in modelos_estand:
            lista_final.append(f"{col}_{modelo}")

    fig, axes = plt.subplots(nrows=len(columnas)*2, ncols=len(modelos_estand)+1, figsize=figsize)
    axes = axes.flat 

    i=0
    for columna in lista_final:
        sns.boxplot(x=columna, data=df, ax=axes[i], flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5, 'linestyle': 'none'})
        i+=1

    for columna in lista_final:
        sns.histplot(x=columna, data=df, ax=axes[i])
        i+=1

    plt.tight_layout()
