import hdbscan
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from tqdm import tqdm
import pickle

"""
File contains the HDBSCAN classes used to generate the HDBSCAN cluster labels
Such labels are used for both sports profiling and UMAP-HDBSCAN clustering
"""


class GenerateHDBSCAN:
    """
    HDBSCAN Class.
    Uses a grid-search approach, optimising for DBCV
    Used for sport clustering analysis

    Class has been adapted from: 
    Frenzel, C. (2025) Tuning with HDBSCAN, Towards Data Science. 
    Available at: https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970/ (Accessed: 27 August 2026). 
    """
    def __init__(self, emb : np.ndarray):
        self.emb = emb
        self.clusterer = None

    def hdb(self):
        optimal_hp = self.hdb_dbcv()
        cluster = hdbscan.HDBSCAN(
            min_samples = optimal_hp['min_samples'],
            min_cluster_size = optimal_hp['min_cluster_size'],
            cluster_selection_method = optimal_hp['cluster_selection_method'],
            metric = optimal_hp['metric']
            ).fit(self.emb)
        self.clusterer = cluster
        return self.get_clusterer()

    def hdb_dbcv(

        self,
        min_samples: list[int] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_cluster_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        cluster_selection_method : list[str]  = ['eom', 'leaf'],
        metric : list[str] = ['euclidean', 'manhattan']) -> pd.DataFrame:
        frames = []
        best = {
            'min_samples': 0,
            'min_cluster_size': 0,
            'cluster_selection_method': 0,
            'metric' : '',
            'num_clusters' : 0,
            'unclustered_prop' : 0,
            'dbcv' : float('-inf')
        }
        for min_sample in tqdm(min_samples):
            for min_cluster in min_cluster_size:
                for cluster_selection in cluster_selection_method:
                    for m in metric:
                        hdb = hdbscan.HDBSCAN(
                            min_samples = min_sample,
                            min_cluster_size = min_cluster,
                            cluster_selection_method = cluster_selection,
                            metric = m,
                            gen_min_span_tree = True
                        ).fit(self.emb)
                        num_clusters = len(set(hdb.labels_))
                        unclustered_prop = len([c for c in hdb.labels_ if c == -1])/len(hdb.labels_)
                        dbcv = hdb.relative_validity_
                        if dbcv > best['dbcv']:
                            best['min_samples'] = min_sample
                            best['min_cluster_size'] = min_cluster
                            best['cluster_selection_method'] = cluster_selection
                            best['metric'] = m
                            best['num_clusters'] = num_clusters
                            best['unclustered_prop'] = unclustered_prop
                            best['dbcv'] = dbcv
                        frames.append([min_sample, min_cluster, cluster_selection, m, num_clusters, unclustered_prop, dbcv])
        df_output = pd.DataFrame(frames, columns = ['min_sample', 'min_cluster', 'cluster_selection','metric', 'num_clusters', 'unclustered_prop', 'dbcv'])
        print(df_output.sort_values('dbcv', ascending=False).head())
        return best
    
    def hdb_narrowed(self):
        optimal_hp = self.hdb_dbcv_narrowed()
        cluster = hdbscan.HDBSCAN(
            min_samples = optimal_hp['min_samples'],
            min_cluster_size = optimal_hp['min_cluster_size'],
            cluster_selection_method = optimal_hp['cluster_selection_method'],
            metric = optimal_hp['metric']
            ).fit(self.emb)
        self.clusterer = cluster
        return self.get_clusterer()

    def hdb_dbcv_narrowed(

        self,
        min_samples: list[int] = [20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_cluster_size : list[int] = [250, 300, 350, 400, 450, 500],
        cluster_selection_method : list[str]  = ['eom'],
        metric : list[str] = ['euclidean']) -> pd.DataFrame:
        frames = []
        best = {
            'min_samples': 0,
            'min_cluster_size': 0,
            'cluster_selection_method': 0,
            'metric' : '',
            'num_clusters' : 0,
            'unclustered_prop' : 0,
            'dbcv' : float('-inf')
        }
        for min_sample in tqdm(min_samples):
            for min_cluster in min_cluster_size:
                for cluster_selection in cluster_selection_method:
                    for m in metric:
                        hdb = hdbscan.HDBSCAN(
                            min_samples = min_sample,
                            min_cluster_size = min_cluster,
                            cluster_selection_method = cluster_selection,
                            metric = m,
                            gen_min_span_tree = True
                        ).fit(self.emb)
                        num_clusters = len(set(hdb.labels_))
                        unclustered_prop = len([c for c in hdb.labels_ if c == -1])/len(hdb.labels_)
                        dbcv = hdb.relative_validity_
                        if dbcv > best['dbcv']:
                            best['min_samples'] = min_sample
                            best['min_cluster_size'] = min_cluster
                            best['cluster_selection_method'] = cluster_selection
                            best['metric'] = m
                            best['num_clusters'] = num_clusters
                            best['unclustered_prop'] = unclustered_prop
                            best['dbcv'] = dbcv
                        frames.append([min_sample, min_cluster, cluster_selection, m, num_clusters, unclustered_prop, dbcv])
        df_output = pd.DataFrame(frames, columns = ['min_sample', 'min_cluster', 'cluster_selection','metric', 'num_clusters', 'unclustered_prop', 'dbcv'])
        print(df_output.sort_values('dbcv', ascending=False).head())
        return best
    
    def get_clusterer(self):
        return self.clusterer

class GenerateHDBSCAN2:
    """
    HDBSCAN2 Class.
    Similar to GenerateHBDSCAN2 however the .fit method takes emb as an input compared to __init__()
    Used for UMAP embedding clustering.

    Class has been adapted from: 
    Frenzel, C. (2025) Tuning with HDBSCAN, Towards Data Science. 
    Available at: https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970/ (Accessed: 27 August 2026). 

    """
    def __init__(self):
        self.model = None
        self.hyperparams = {
            'min_samples' : None,
            'min_cluster_size' : None,
            'cluster_selection_method' : None,
            'metric' : None,
            'core_dist_n_jobs' : -1
        }
        self.min_samples = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.min_cluster_size = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        self.cluster_selection_method = ['eom', 'leaf']
        self.metric = ['euclidean', 'manhattan']
        self.name = self.__class__.__name__
        self.save_path = ROOT / 'src' / 'stream_3' / 'saved_models'
        self.output_df = None
        self.dbcv = None

    def fit(self, emb : np.array):
        print('Tuning HDBSCAN hyperparameters')
        self.tune_hyperparams(emb)
        self.model = hdbscan.HDBSCAN(**self.hyperparams).fit(emb)
        print('HBDCSAN fit successfully.')
        # self.save_class()

    def tune_hyperparams(self, emb : np.array):
        frames = []
        best = {
            'min_samples': 0,
            'min_cluster_size': 0,
            'cluster_selection_method': 0,
            'metric' : '',
            'num_clusters' : 0,
            'unclustered_prop' : 0,
            'dbcv' : float('-inf')
        }
        for min_sample in tqdm(self.min_samples):
            for min_cluster_size in self.min_cluster_size:
                for cluster_selection_method in self.cluster_selection_method:
                    for metric in self.metric:
                        hdb = hdbscan.HDBSCAN(
                            min_samples = min_sample,
                            min_cluster_size = min_cluster_size,
                            cluster_selection_method = cluster_selection_method,
                            metric = metric,
                            gen_min_span_tree = True,
                            core_dist_n_jobs = -1
                        ).fit(emb) 
                        num_clusters = len(set(hdb.labels_))
                        unclustered_prop = len([c for c in hdb.labels_ if c == -1])/len(hdb.labels_)
                        dbcv = hdb.relative_validity_
                        if dbcv > best['dbcv']:
                            best['min_samples'] = min_sample
                            best['min_cluster_size'] = min_cluster_size
                            best['cluster_selection_method'] = cluster_selection_method
                            best['metric'] = metric
                            best['num_clusters'] = num_clusters
                            best['unclustered_prop'] = unclustered_prop
                            best['dbcv'] = dbcv
                        frames.append([min_sample, min_cluster_size, cluster_selection_method, metric, num_clusters, unclustered_prop, dbcv])
        df_output = pd.DataFrame(frames, columns = ['min_sample', 'min_cluster', 'cluster_selection','metric', 'num_clusters', 'unclustered_prop', 'dbcv'])
        print(df_output.sort_values('dbcv', ascending=False).head(10))
        self.hyperparams['min_samples'] = best['min_samples']
        self.hyperparams['min_cluster_size'] = best['min_cluster_size']
        self.hyperparams['cluster_selection_method'] = best['cluster_selection_method']
        self.hyperparams['metric'] = best['metric']
        self.hyperparams['gen_min_span_tree'] = True
        self.output_df = df_output
        self.dbcv = best['dbcv']

    def get_model(self):
        return self.model
    def get_condensed_tree(self):
        return self.model.condensed_tree_.plot(select_clusters=True)
    def get_single_linkage_tree(self):
        return self.model.single_linkage_tree_.plot()

    def save_class(self):
        filename = f'{self.__class__.__name__}.sav'
        path = self.save_path / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.__class__.__name__} saved to: {path}')
    def get_df_output(self):
        return self.output_df
    def get_dbcv(self):
        return self.dbcv