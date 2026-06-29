import hdbscan 
from tqdm import tqdm
import pandas as pd
import numpy as np

def hdb_dbcv(
        emb : np.ndarray,
        min_samples: list[int] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_cluster_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        cluster_selection_method : list[str]  = ['eom', 'leaf'],
        metric : list[str] = ['euclidean', 'manhattan']) -> pd.DataFrame:
    outputs = []
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
                    ).fit(emb)
                    num_clusters = len(set(hdb.labels_))
                    unclustered_prop = len([c for c in hdb.labels_ if c == -1])/len(hdb.labels_)
                    dbcv = hdb.relative_validity_
                    outputs.append([min_sample, min_cluster, cluster_selection, m, num_clusters, unclustered_prop, dbcv])
    df_output = pd.DataFrame(outputs, columns = ['min_sample', 'min_cluster', 'cluster_selection', 'm', 'num_clusters', 'unclustered_prop', 'dbcv'])
    print(df_output.sort_values('dbcv', ascending=False).head())
    return df_output