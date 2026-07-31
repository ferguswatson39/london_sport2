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
from src.loading_data.data_catalogue import DataCatalogue


if __name__ == '__main__':

    arr_path = ROOT / 'src' / 'stream_3' / 'embeddings' / # Enter umap emb .npy file
    save_path = ROOT / 'src' / 'stream_3' / 'cluster_labels' 
    data_path = ROOT / 'data' / # Enter clustering dataset path
    arr = np.load(arr_path)
    df = pd.read_csv(data_path)

    dc = DataCatalogue()
    continuous_cols = dc.get_clustering_continuous()
    categorical_cols = dc.get_clustering_categoricals()


    models = [GenerateKMEANS(), GenerateKprototypes(), GenerateHDBSCAN2()]
    for model in models:
        model_name = model.__class__.__name__
        if model_name != 'GenerateKprototypes' :
            model.fit(arr)
        else:
            scaler = StandardScaler()
            df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
            model.fit(df, categorical_cols)

        labels = model.get_model().labels_
        np.save(save_path / f'{model_name}_labels.npy', labels)

        
            
