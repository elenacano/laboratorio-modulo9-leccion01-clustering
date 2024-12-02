# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Para la visualización 
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Otros objetivos
# -----------------------------------------------------------------------
import math
from itertools import combinations


# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Para la codificación de las variables numéricas
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder # para poder aplicar los métodos de OneHot, Ordinal,  Label y Target Encoder 
from category_encoders import TargetEncoder # type: ignore


def one_hot_encoding(dataframe, diccionario_encoding,):
        """
        Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.

        Returns:
            - dataframe: DataFrame de pandas, el DataFrame con codificación one-hot aplicada.
        """
        # accedemos a la clave de 'onehot' para poder extraer las columnas a las que que queramos aplicar OneHot Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = diccionario_encoding.get("onehot", [])

        # si hay contenido en la lista 
        if col_encode:

            one_hot_encoder = OneHotEncoder()
            trans_one_hot = one_hot_encoder.fit_transform(dataframe[col_encode])
            oh_df = pd.DataFrame(trans_one_hot.toarray(), columns=one_hot_encoder.get_feature_names_out())
            dataframe = pd.concat([dataframe.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)


            required_columns = [
                'size', 'municipality', 'distance', 'floor', 'hasLift',
                'rooms_0', 'rooms_1', 'rooms_2', 'rooms_3', 'rooms_4',
                'bathrooms_1', 'bathrooms_2', 'bathrooms_3',
                'propertyType_chalet', 'propertyType_countryHouse', 'propertyType_duplex',
                'propertyType_flat', 'propertyType_penthouse', 'propertyType_studio',
                'exterior_True'
            ]

            missing_columns = [col for col in required_columns if col not in dataframe.columns]

            # Añadir las columnas faltantes con valor 0
            for col in missing_columns:
                dataframe[col] = 0

            dataframe = dataframe[required_columns]

    
        return dataframe