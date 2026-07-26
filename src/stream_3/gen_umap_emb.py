import umap
#import umap.plot
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.loading_data.data_catalogue import DataCatalogue
from src.stream_3.gower_distance import get_gower

class GenerateUmapEmb:
    """
    Generates n dimensional umap embedding
    Splits data into continuous and categorical and applied separate umap models using euclidean and dice distance respectively
    Then performs a fuzzy set intersection which combined the two fuzzy graphs then re-embeds using the combined graph
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.dc = DataCatalogue()
        self.df_cols = list(self.df.columns)
        self.scaler = None
        self.categorical_umap = None
        self.continuous_umap = None
        self.intersection_umap = None
        self.encoder = None
        self.gower_umap = None
        self.sports_umap = None

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

    def fit_umap(self, num_dimensions : int = 2):
        avail_cats, avail_contins = self.gen_splits()
        X_scaled_continuous = self.scale_continuous(avail_contins)
        print(f'Continuous Shape: {X_scaled_continuous.shape}')
        X_categoricals = self.dummy_encode(avail_cats)
        print(f'Categorical Shape: {X_categoricals.shape}')
        assert X_scaled_continuous.shape[0] == X_categoricals.shape[0] == len(self.df)
        continuous = umap.UMAP(min_dist = 0.0, n_components = num_dimensions, n_neighbors = 15, metric = 'euclidean', random_state=42).fit(X_scaled_continuous)
        print('Finished fitting continuous....')
        categorical = umap.UMAP(min_dist = 0.0, n_components = num_dimensions, n_neighbors = 100, metric = 'dice', random_state=42).fit(X_categoricals)
        print('Finished fitting categorical....')
        intersection = continuous * categorical
        print('Finished computing intersection...')
        self.continuous_umap = continuous
        self.categorical_umap = categorical
        self.intersection_umap = intersection
        print('UMAP fit complete!')
        return self.get_intersection_umap()
    
    def fit_umap_using_gower(self, num_dimensions : int = 2):
        catogs = ['Eth7', 'Gend3', 'HHLiv12', 'WorkStat10']
        contins = ['Age9', 'Educ6', 'IMD10', 'NSSEC5', 'Motiva_POP', 'motivd_POP', 'happy', 'lifesat', 'lone', 'worthw', 'comm1', 'inclus_a']
        distances = get_gower(self.df[catogs + contins], catogs)
        emb = umap.UMAP(min_dist = 0.1, n_components = num_dimensions, n_neighbors = 25, random_state = 42).fit(distances)
        self.gower_umap = emb
        return self.get_gower_umap()
    
    def fit_umap_sports(self, metric : str = 'jaccard', num_neighbours : int = 25, num_dimensions : int = 2):
        self.sports_umap = umap.UMAP(min_dist = 0.1, n_components = num_dimensions, n_neighbors = num_neighbours, random_state = 42, metric = metric, init = 'random').fit(self.df.values)
        return self.sports_umap

    def get_categorical_umap(self):
        return self.categorical_umap
    def get_continuous_umap(self):
        return self.continuous_umap
    def get_intersection_umap(self):
        return self.intersection_umap
    def get_scaler(self):
        return self.scaler
    def get_encoder(self):
        return self.encoder
    def get_gower_umap(self):
        return self.gower_umap
    def get_sports_umap(self):
        return self.sports_umap
    

