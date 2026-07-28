from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm

class GenerateKMEANS:
    def __init__(self):
        self.model = None
        self.hyperparams = {
            'n_clusters' : None,
            'random_state' : 42,
            'n_init' : 10,
            # Discuss this for write up
            'init' : 'k-means++'
        }

    def fit(self, emb : np.array):
        self.tune_hyperparams(emb)
        print('K-Means fit successfully.')

    def tune_hyperparams(self, emb : np.array):
        best_results = {
            'best_k' : None,
            'best_silh' : float('-inf'),
            'best_model' : None
        }
        frames = []
        for i in tqdm(range(2, 50)):
            k = KMeans(
                n_clusters = i,
                random_state = self.hyperparams['random_state'],
                n_init = self.hyperparams['n_init'],
                init = self.hyperparams['init']
            ).fit(emb)
            silh = silhouette_score(emb, k.labels_)
            if best_results['best_silh'] < silh:
                best_results['best_silh'] = silh
                best_results['best_k'] = i
                best_results['best_model'] = k
            frames.append([i, silh])
        output = pd.DataFrame(frames, columns = ['num_K', 'silh_score'])
        print(output.sort_values('silh_score', ascending=False).head())
        self.hyperparams['n_clusters'] = best_results['best_k']
        self.model = best_results['best_model']
    def get_model(self):
        return self.model





