import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.loading_data.data_catalogue import DataCatalogue
from src.loading_data.load_data import get_clean_2022, one_hot_encode_frame


def preprocess_perturbation(model : object, cluster_column : str, target : str):

    if model.__class__.__name__ in ['OLSRegressor','Logistic']:
        drop = 'first'
    else:
        drop = None

    dc = DataCatalogue()
    df = get_clean_2022()

    continuous_vars = dc.get_perturbation_core_contins() + dc.get_perturbation_vars()

    df['MEMS7_ALL'] = df['MEMS7_ALL'].clip(upper=6720)
    df['MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])

    to_encode_vars = dc.get_perturbation_core_to_encode()
    assert to_encode_vars == ['Gend3', 'Eth7', 'WorkStat8', 'HHLiv9']

    df_encoded = one_hot_encode_frame(df, to_encode_vars, drop)

    drop_cols = ['serial', 'year', 'LCA_Class', 'MEMS7_ALL', 'active']
    keep_cols = [col for col in df_encoded.columns if col not in drop_cols]
    print(f'Keep columns : {keep_cols}')

    labels = df_encoded[cluster_column]
    Y = df_encoded[target]
    X = df_encoded[keep_cols]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, cluster_column]

    scaler = StandardScaler()
    X_train[continuous_vars] = scaler.fit_transform(X_train[continuous_vars])
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])
    
    return X_train, X_test, Y_train, Y_test, test_set_clusters, scaler

if __name__ == '__main__':
    preprocess_perturbation('ye', 'LCA_Class', 'MEMS7_ALL')