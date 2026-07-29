import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.stream_3.gen_kmean_clus import GenerateKMEANS
from src.stream_3.gen_kprototypes_clus import GenerateKprototypes
from src.stream_3.gen_hdb_clus import GenerateHDBSCAN2



def create_clustering_labels(input_df : pd.DataFrame, categorical_cols : list[str]) -> pd.DataFrame:
    models = [GenerateKMEANS(), GenerateKprototypes(), GenerateHDBSCAN2()]
    df = input_df.copy()
    output_df = input_df.copy()

    for model in models:

        process(df, model, categorical_cols, continuous_cols)
        labels = model.get_model().labels_
        output_df[f'{model.__class__.__name__}_labels'] = labels
    return output_df

def process(df : pd.DataFrame, model : object, categorical_cols : list[str], continuous_cols : list[str]):
    scaler = StandardScaler()
    if model.__class__.__name__ == 'GenerateKprototypes':

        """
        scale continuous vars for k prototypes
        """
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
        model.fit(df)
    else:
        encoder = OneHotEncoder(sparse_output = False).set_output(transform = 'pandas')
        categoricals = df[categorical_cols]
        encoded_cats = encoder.fit_transform(categoricals)
        df_encoded = pd.concat([df, encoded_cats], axis = 1).drop(columns = categorical_cols)
        X_scaled = scaler.fit_transform(df_encoded)
        model.fit(X_scaled)



