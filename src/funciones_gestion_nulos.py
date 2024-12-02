import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


def imputar_nulos_numericos(df, cols_numericas, imputer, neighbors=5):

    if imputer not in ["KNNImputer", "IterativeImputer"]:
        print("ERROR: imputador no válido.")
        return
    
    df_num = df.select_dtypes(include=np.number)
    df_num_sin_VR = df_num[cols_numericas]

    if imputer == "KNNImputer":
        imputador = KNNImputer(n_neighbors = neighbors)

    elif imputer == "IterativeImputer":
        imputador = IterativeImputer(estimator=RandomForestRegressor())


    df_imputado = imputador.fit_transform(df_num_sin_VR) #columnas que quiero que use para rellenar los vecinos
    df_num_sin_nulos = pd.DataFrame(df_imputado, columns=df_num_sin_VR.columns)

    df_imputer = df.copy()
    df_imputer[df_num_sin_nulos.columns]=df_num_sin_nulos

    return df_imputer


def imputar_nulos_categoricos(df, cols_categoricas, imputer):

    df[cols_categoricas] = df[cols_categoricas].fillna(imputer)

    return

