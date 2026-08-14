from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from src.loading_data.load_data import get_master_clustering_input
from src.loading_data.data_catalogue import DataCatalogue

class GenerateKPrototypes:
    def __init__(self):
        self.model = None 
        self.hyperparams = {
            'n_clusters' : None,
            'init' : None,
            'n_init' : 10,
            'random_state' : 42,
        }
        self.init = ['Cao', 'Huang']
        self.n_clusters = range(2, 50)
        self.save_path = ROOT / 'src' / 'stream_3' / 'saved_models'

    def fit(self, emb : np.array, categorical_cols : list[int]):
        self.tune_hyperparams(emb, categorical_cols)
        print('K-Prototypes fit successfully')
        self.save_class()

    def tune_hyperparams(self, emb : np.array, categorical_cols : list[int]):
        best_results = {
            'best_k' : None,
            'best_silh' : float('-inf'),
            'best_model' : None, 
            'best_init' : None
        }
        frames = []
        for n_clusters in tqdm(self.n_clusters):
            for init in self.init:
                k = KPrototypes(
                    n_clusters = n_clusters,
                    n_init = 10, 
                    init = init, 
                    random_state = 42).fit(emb, categorical=categorical_cols)
                silh = silhouette_score(emb, k.labels_)
                if best_results['best_silh'] < silh:
                    best_results['best_silh'] = silh
                    best_results['best_k'] = n_clusters
                    best_results['best_model'] = k
                    best_results['best_init'] = init
                frames.append([n_clusters, silh, init])
        output = pd.DataFrame(frames, columns = ['num_K', 'silh_score', 'init'])
        print(output.sort_values('silh_score', ascending=False).head())
        self.hyperparams['n_clusters'] = best_results['best_k']
        self.hyperparams['init'] = best_results['best_init']
        self.model = best_results['best_model']

    def save_class(self):
        filename = f'{self.__class__.__name__}.sav'
        path = self.save_path / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.__class__.__name__} saved to: {path}')

    def get_model(self):
        return self.model


class GenerateKPrototypes2:
    def __init__(self):
        self.cost = []
        self.save_path = ROOT / 'src' / 'stream_3' / 'k_prototypes'
        if self.save_path.exists():
            print('K-prototypes save path found')
        else:
            raise FileNotFoundError('Save path not found for k_prototypes')

    def fit(self, arr : np.array, categorical_idx : list[int]):
        for n in tqdm(range(2, 80)):
            k_proto = KPrototypes(n_clusters=n, n_jobs=-1, random_state=42)
            k_proto.fit_predict(arr, categorical = categorical_idx)
            self.cost.append(k_proto.cost_)

    def get_costs(self):
        return self.cost
    def get_silh_score(self):
        return self.silh_score
    def save_class(self):
        filename = f'{self.__class__.__name__}.sav'
        path = self.save_path / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.__class__.__name__} saved to: {path}')


if __name__ == '__main__':
    df = get_master_clustering_input()
    dc = DataCatalogue()
    categoricals = dc.get_clustering_categoricals()
    print(f'Categoricals: {categoricals}')
    categorical_idx = [df.columns.get_loc(col) for col in categoricals]

    k_proto = GenerateKPrototypes2()
    k_proto.fit(df.values, categorical_idx)
    k_proto.save_class()
