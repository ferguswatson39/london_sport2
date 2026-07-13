from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


class GenerateKMEANS:
    def __init__(self, emb : np.ndarray):
        self.emb = emb

    def kmeans(self):
        optimised_k = self.kmeans_cv()
        k = KMeans(n_clusters = optimised_k['best_k'], random_state = 42, n_init='auto').fit(self.emb)
        return k.labels_
    
    def kmeans_cv(self):
        results = {
            'best_k' : 0,
            'best_silh' : float('-inf'),
        }
        for i in range(2, 50):
            k = KMeans(n_clusters = i, random_state = 42, n_init='auto').fit(self.emb)
            silh = silhouette_score(self.emb, k.labels_)
            if results['best_silh'] < silh:
                results['best_silh'] = silh
                results['best_k'] = i
        return results