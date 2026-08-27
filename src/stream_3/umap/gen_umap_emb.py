import umap
import umap.plot
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
import pickle
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.loading_data.data_catalogue import DataCatalogue

class GenerateUmapEmb:
    """
    Generates n dimensional umap embedding
    This UMAP class was used for the sports clustering pipeline
    
    Adapted from: 
        McInnes, L. (2018):
        Combining multiple UMAP models - umap 0.5.8 documentation. 
        Available at: https://umap-learn.readthedocs.io/en/latest/composing_models.html (Accessed: 27 August 2026). 

    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.dc = DataCatalogue()
        self.df_cols = list(self.df.columns)
        self.scaler = None
        self.categorical_umap = None
        self.continuous_umap = None
        self.intersection_umap = None
        self.euclidean_umap = None
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
        return X_categoricals

    def fit_fuzzy_umap(self, num_dimensions : int = 2):
        avail_cats, avail_contins = self.gen_splits()
        X_scaled_continuous = self.scale_continuous(avail_contins)
        print(f'Continuous Shape: {X_scaled_continuous.shape}')
        X_categoricals = self.dummy_encode(avail_cats)
        print(f'Categorical Shape: {X_categoricals.shape}')
        assert X_scaled_continuous.shape[0] == X_categoricals.shape[0] == len(self.df)
        continuous = umap.UMAP(min_dist = 0.0, n_components = num_dimensions, n_neighbors = 500, metric = 'euclidean').fit(X_scaled_continuous)
        print('Finished fitting continuous....')
        categorical = umap.UMAP(min_dist = 0.0, n_components = num_dimensions, n_neighbors = 500, metric = 'dice').fit(X_categoricals.values)
        print('Finished fitting categorical....')
        intersection = continuous + categorical
        print('Finished computing intersection...')
        self.continuous_umap = continuous
        self.categorical_umap = categorical
        self.intersection_umap = intersection
        print('UMAP fit complete!')
        return self.get_intersection_umap()

    def fit_umap_using_euclidean(self, num_dimensions : int = 2):
        categoricals = ['Gend3', 'Eth7', 'Disab2_POP', 'WorkStat8', 'HHLiv9']
        contins = ['Age9', 'Educ6', 'NSSEC5', 'IMD10', 'Child4', 'motivd_POP', 'Motiva_POP']
        scaler = StandardScaler()
        X_categoricals = self.dummy_encode(categoricals)
        X_contins = self.df[contins]
        combined = pd.concat([X_categoricals, X_contins], axis=1)
        X_scaled = scaler.fit_transform(combined)
        self.euclidean_umap = umap.UMAP(min_dist=0.1, n_components = num_dimensions,n_neighbors = 20, metric='euclidean').fit(X_scaled)
        return self.get_euclidean_umap()

    def fit_umap_sports(self, metric : str = 'jaccard', num_neighbours : int = 25, num_dimensions : int = 2):
        self.sports_umap = umap.UMAP(min_dist = 0.1, n_components = num_dimensions, n_neighbors = num_neighbours, random_state = 42, metric = metric, init = 'random').fit(self.df.values)
        return self.sports_umap

    def get_categorical_umap(self):
        return self.categorical_umap
    def get_continuous_umap(self):
        return self.continuous_umap
    def get_intersection_umap(self):
        return self.intersection_umap
    def get_euclidean_umap(self):
        return self.euclidean_umap
    def get_scaler(self):
        return self.scaler
    def get_encoder(self):
        return self.encoder
    def get_gower_umap(self):
        return self.gower_umap
    def get_sports_umap(self):
        return self.sports_umap
    

class GenerateUmapEmb2:
    """
    Generates n dimensional umap embedding
    Splits data into continuous and categorical and applied separate umap models using euclidean and jaccard distance respectively
    Then performs a fuzzy set intersection then re-embeds using the combined graph
    
    This UMAP class was used for generating the UMAP embedding for the UMAP-HDBSCAN clustering pipeline

    Adapted from: 
        McInnes, L. (2018):
        Combining multiple UMAP models - umap 0.5.8 documentation. 
        Available at: https://umap-learn.readthedocs.io/en/latest/composing_models.html (Accessed: 27 August 2026). 

    """

    def __init__(self):

        self.dc = DataCatalogue()
        self.scaler = None
        self.categorical_umap = None
        self.continuous_umap = None
        self.intersection_umap = None
        self.encoder = None
        self.sports_umap = None
        self.emb = None
        self.save_path = ROOT / 'src' / 'stream_3' / 'embeddings'
        if self.save_path.exists():
            print('Save path found.')
        else:
            raise FileNotFoundError(f'{self.save_path} not found.')

    def scale_continuous(self, df : pd.DataFrame, avail_contins : list[str]):
        scaler = StandardScaler()
        X = df[avail_contins]
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        return X_scaled

    def dummy_encode(self, df : pd.DataFrame, avail_cats : list[str]):
        encoder = OneHotEncoder(sparse_output = True)
        categoricals = df[avail_cats]
        X_categoricals = encoder.fit_transform(categoricals)
        self.encoder = encoder
        return X_categoricals

    def fit_fuzzy_umap(self,
                        df : pd.DataFrame,
                        avail_contins : list[str],
                        avail_cats : list[str],
                        continuous_neighbors : int,
                        categorical_neighbors : int,
                        continuous_metric : str,
                        categorical_metric : str,
                        operator : str,
                        num_dimensions : int = 2):

        if operator not in ['+', '*']:
            raise ValueError(f'{operator} | not a valid operator')
        
        X_scaled_continuous = self.scale_continuous(df, avail_contins)
        print(f'Continuous Shape: {X_scaled_continuous.shape}')

        X_categoricals = self.dummy_encode(df, avail_cats)
        print(f'Categorical Shape: {X_categoricals.shape}')

        assert X_scaled_continuous.shape[0] == X_categoricals.shape[0] == len(df)

        continuous = umap.UMAP(min_dist = 0.0, n_components = num_dimensions, n_neighbors = continuous_neighbors, metric = continuous_metric, random_state=42).fit(X_scaled_continuous)
        print('Finished fitting continuous....')

        categorical = umap.UMAP(min_dist = 0.0, n_components = num_dimensions, n_neighbors = categorical_neighbors, metric = categorical_metric, random_state=42).fit(X_categoricals)
        print('Finished fitting categorical...')

        print('Computing fuzzy set intersection...')
        if operator == '*':
            intersection = continuous * categorical
            method = 'multip'
        elif operator == '+':
            intersection = continuous + categorical
            method = 'plus'
        print('Finished computing intersection...')

        self.continuous_umap = continuous
        self.categorical_umap = categorical
        self.intersection_umap = intersection
        print('UMAP fit complete!')
        self.emb = self.get_intersection_umap().embedding_
        np.save(self.save_path / f'EMB_UMAP_conn_{continuous_neighbors}_catn_{categorical_neighbors}_conm_{continuous_metric}_catm_{categorical_metric}_{method}.npy', self.emb)
        print(f'Embedding saved to\n {self.save_path}')

        self.save_class(continuous_neighbors, categorical_neighbors, continuous_metric, categorical_metric, method)

        img = umap.plot.points(self.intersection_umap)
        img_path = ROOT / 'figures' / 'umap' / f'IMAGE_UMAP_conn_{continuous_neighbors}_catn_{categorical_neighbors}_conm_{continuous_metric}_catm_{categorical_metric}_{method}.png'
        plt.savefig(img_path)
        plt.close()

        return self.get_intersection_umap()

    def fit_umap_sports(self, df : pd.DataFrame, metric : str = 'jaccard', num_neighbours : int = 25, num_dimensions : int = 2):
        self.sports_umap = umap.UMAP(min_dist = 0.1, n_components = num_dimensions, n_neighbors = num_neighbours, random_state = 42, metric = metric, init = 'random').fit(df.values)
        return self.get_sports_umap()

    def get_intersection_umap(self):
        return self.intersection_umap

    def get_sports_umap(self):
        return self.sports_umap

    def save_class(self, continuous_neighbors : str, categorical_neighbors : str, continuous_metric : str, categorical_metric : str, method : str):
        filename = f'CLASS_UMAP_conn_{continuous_neighbors}_catn_{categorical_neighbors}_conm_{continuous_metric}_catm_{categorical_metric}_{method}.npy'
        path = self.save_path / 'classes' / filename
        with open(path, 'wb') as file:
            pickle.dump(self.intersection_umap, file)
        print(f'{filename} saved to: {path}')
    
