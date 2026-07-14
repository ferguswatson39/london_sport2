import umap
import umap.plot
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.loading_data.data_catalogue import DataCatalogue

class GenerateUmapEmb:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.dc = DataCatalogue()
        self.df_cols = list(df.columns)
        self.scaler = None
        self.categorical_umap = None
        self.continuous_umap = None
        self.intersection_umap = None
        self.encoder = None

    def gen_splits(self):
        avail_cats = [c for c in self.df_cols if c in self.dc.get_clustering_categoricals()]
        avail_contins = [c for c in self.df_cols if c in self.dc.get_clustering_continuous()]
        print(f'Categorical Columns for UMAP:\n {avail_cats}')
        print(f'Continuous Columns for UMAP\n{avail_contins}')
        return avail_cats, avail_contins
    def scale_continuous(self, avail_contins : list[str]):
        standard_scaler = StandardScaler()
        X = self.df[avail_contins].values
        X_scaled = standard_scaler.fit_transform(X)
        self.scaler = standard_scaler
        return X_scaled

    def dummy_encode(self, avail_cats : list[str]):
        encoder = OneHotEncoder(sparse_output = False).set_output(transform = 'pandas')
        categoricals = self.df[avail_cats]
        X_categoricals = encoder.fit_transform(categoricals)
        self.encoder = encoder
        return X_categoricals.values

    def fit_umap(self):
        avail_cats, avail_contins = self.gen_splits()
        X_scaled_continuous = self.scale_continuous(avail_contins)
        X_categoricals = self.dummy_encode(avail_cats)
        continuous = umap.UMAP(n_neighbors = 15, metric = 'euclidean', random_state = 42).fit(X_scaled_continuous)
        categorical = umap.UMAP(n_neighbors = 100, metric = 'dice', random_state = 42).fit(X_categoricals)
        intersection = continuous * categorical
        self.continuous_umap = continuous
        self.categorical_umap = categorical
        self.intersection_umap = intersection
        return intersection
    
    
    def get_categorical_umap(self):
        return self.categorical_umap
    def get_continuous_umap(self):
        return self.continuous_umap
    def get_scaler(self):
        return self.scaler
    def get_encoder(self):
        return self.encoder
    

