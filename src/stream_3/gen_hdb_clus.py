import hdbscan
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from tqdm import tqdm
import gower

class GenerateHBDSCAN:
    """
    HDBSCAN Class.
    Requires data to be input in the form of a gower matrix
    Can change to euclidean but would need to change metric
    """
    def __init__(self, emb : np.ndarray):
        self.emb = emb

    def hdb(self):
        optimal_hp = self.hdb_dbcv()
        clusters = hdbscan.HDBSCAN(
            min_samples = optimal_hp['min_samples'],
            min_cluster_size = optimal_hp['min_cluster_size'],
            cluster_selection_method = optimal_hp['cluster_selection_method'],
            metric = 'precomputed'
            ).fit(self.emb)
        return clusters.labels_

    def hdb_dbcv(
        ## Adapted From: https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970/
        self,
        min_samples: list[int] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_cluster_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        cluster_selection_method : list[str]  = ['eom', 'leaf']) -> pd.DataFrame:
        frames = []
        best = {
            'min_samples': 0,
            'min_cluster_size': 0,
            'cluster_selection_method': 0,
            'num_clusters' : 0,
            'unclustered_prop' : 0,
            'dbcv' : float('-inf')
        }
        for min_sample in tqdm(min_samples):
            for min_cluster in min_cluster_size:
                for cluster_selection in cluster_selection_method:
                    hdb = hdbscan.HDBSCAN(
                        min_samples = min_sample,
                        min_cluster_size = min_cluster,
                        cluster_selection_method = cluster_selection,
                        metric = 'precomputed',
                        gen_min_span_tree = True
                    ).fit(self.emb)
                    num_clusters = len(set(hdb.labels_))
                    unclustered_prop = len([c for c in hdb.labels_ if c == -1])/len(hdb.labels_)
                    dbcv = hdb.relative_validity_
                    if dbcv > best['dbcv']:
                        best['min_samples'] = min_sample
                        best['min_cluster_size'] = min_cluster
                        best['cluster_selection_method'] = cluster_selection
                        best['num_clusters'] = num_clusters
                        best['unclustered_prop'] = unclustered_prop
                        best['dbcv'] = dbcv
                    frames.append([min_sample, min_cluster, cluster_selection, num_clusters, unclustered_prop, dbcv])
        print(f'Optimal HDBSCAN Hyperparameters:\n>>>{best}')
        df_output = pd.DataFrame(frames, columns = ['min_sample', 'min_cluster', 'cluster_selection', 'num_clusters', 'unclustered_prop', 'dbcv'])
        print(df_output.sort_values('dbcv', ascending=False).head())
        return best
