import pandas as pd # type: ignore
import numpy as np
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

import math

import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------------------------------------
#                                    EXPLORACIÓN DATAFRAME
#-------------------------------------------------------------------------------------------------

def exploracion_dataframe(dataframe):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0]) # type: ignore
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"])) # type: ignore
    
    print("\n ..................... \n")

    display(dataframe.describe().T)
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())     # type: ignore

def display_value_count(df):
    columnas = df.columns
    for columna in columnas:
        display(df[columna].value_counts())

def separar_df(df):
    return df.select_dtypes(include=np.number), df.select_dtypes(include="O")


def plot_numericas(df, figsize=(15,10)):
    columnas_numericas = df.columns
    num_filas = math.ceil(len(columnas_numericas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):
        sns.histplot(df, x=columna, ax=axes[indice], bins=50)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(columnas_numericas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()




def plot_categoricas(df, figsize=(15,20)):
    columnas_categoticas = df.columns
    num_filas = math.ceil(len(columnas_categoticas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(columnas_categoticas):
        sns.countplot(df, 
                      x=columna, 
                      palette="mako",
                      order = df[columna].value_counts().index,
                      ax=axes[indice])
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")
        axes[indice].tick_params(axis='x', rotation=45)

    if len(columnas_categoticas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()



def relacion_vr_categoricas(df, variable_respuesta):

    df_cat = separar_df(df)[1]
    columnas_categoticas = df_cat.columns
    num_filas = math.ceil(len(columnas_categoticas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=(15,10))
    axes = axes.flat

    for indice, columna in enumerate(columnas_categoticas):

        datos_agrupados = df.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta, ascending=False)

        sns.barplot(datos_agrupados, 
                    x=columna,
                    y=variable_respuesta,
                    palette="mako",
                    ax=axes[indice])
        
        axes[indice].set_title(f"Relación entre {columna} y {variable_respuesta}")
        axes[indice].set_xlabel(f"{columna}")
        axes[indice].tick_params(axis='x', rotation=45)

    if len(columnas_categoticas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()


def relacion_vr_categoricas_problema_categorico(df, vr, figsize=(15,20), rotate=True):
    df_cat = separar_df(df)[1]
    columnas_categoticas = df_cat.columns
    df_cat[vr] = df[vr]
    num_filas = math.ceil(len(columnas_categoticas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(columnas_categoticas):
        sns.countplot(df, 
                      y=columna, 
                      palette="Set1",
                      order = df[columna].value_counts().index,
                      hue=vr,
                      ax=axes[indice])
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")
        if rotate:
            axes[indice].tick_params(axis='x', rotation=45)

    if len(columnas_categoticas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()



def relacion_vr_numericas(df, variable_respuesta):

    df_num = separar_df(df)[0]
    columnas_num = df_num.columns
    num_filas = math.ceil(len(columnas_num)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=(15,10))
    axes = axes.flat

    for indice, columna in enumerate(columnas_num):
        if columna==variable_respuesta:
            fig.delaxes(axes[indice])
            pass
        else:
            sns.scatterplot(df_num, 
                            x=columna,
                            y=variable_respuesta,
                            palette="mako",
                            ax=axes[indice])
            
            axes[indice].set_title(f"Relación entre {columna} y {variable_respuesta}")
            axes[indice].set_xlabel(f"{columna}")
            #axes[indice].tick_params(axis='x', rotation=45)

    if len(columnas_num) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()


def relacion_vr_numericas_problema_categorico(df, vr):
    df_num, df_cat = separar_df(df)
    columnas_numericas = df_num.columns
    num_filas = math.ceil(len(columnas_numericas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=(15,10))
    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):
        sns.histplot(df, x=columna, ax=axes[indice], hue=vr , bins=20)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(columnas_numericas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()


# ---------------------------------- Gráfico de correlación ------------------------------------
def heatmap_correlacion(df):
    df_corr = df.corr(numeric_only=True)
    mascara = np.triu(np.ones_like(df_corr, dtype=np.bool))

    plt.figure(figsize=(10,7))
    sns.heatmap(df_corr, annot=True, vmin=-1, vmax=1, mask=mascara, cmap="coolwarm")


# -------------------------Detección de outliers mediante boxplot ------------------------------------
def detectar_outliers(df, figsize=(15,10), rotate=True):
    
    df_num =separar_df(df)[0]

    columnas_numericas = df_num.columns
    num_filas = math.ceil(len(columnas_numericas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):

        sns.boxplot(df_num, 
                    x=columna,
                    flierprops={"markersize":4, "markerfacecolor":"red"},
                    ax=axes[indice])
        
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")
        if rotate:
            axes[indice].tick_params(axis='x', rotation=45)


    if len(columnas_numericas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()


def detectar_outliers_problema_clasificacion(df, vr, figsize=(15,10), rotate=True):
    
    df_num =separar_df(df)[0]
    columnas_numericas = df_num.columns
    df_num[vr]=df[vr]

    
    num_filas = math.ceil(len(columnas_numericas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):

        sns.boxplot(df_num, 
                    y=columna,
                    flierprops={"markersize":4, "markerfacecolor":"red"},
                    hue=vr,
                    ax=axes[indice])
        
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")
        if rotate:
            axes[indice].tick_params(axis='x', rotation=45)


    if len(columnas_numericas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()




