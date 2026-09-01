import umap
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from tqdm import tqdm
from src.loading_data.data_catalogue import DataCatalogue
from src.loading_data.load_data import get_master_clustering_input
from src.stream_3.umap.gen_umap_emb import GenerateUmapEmb

"""
Pipeline used to generate the 25 UMAP embeddings for the UMAP-HDBSCAN clustering pipeline. 
Searches through n_neighbors search space logarithmically using .fit_fuzzy_umap() create and saves embedding
"""

if __name__ == '__main__':
    dc = DataCatalogue()
    df = get_master_clustering_input()
    emb_generator = GenerateUmapEmb()

    clustering_categoricals = dc.get_clustering_categoricals()
    clustering_continuous = dc.get_clustering_continuous()

    categorical_neighbors, continuous_neighbors = np.round(np.logspace(np.log10(50), np.log10(500), 5)).astype(int) , np.round(np.logspace(np.log10(50), np.log10(500), 5)).astype(int)
    i = 0
    for cat_neighbor in tqdm(categorical_neighbors):
        for contin_neighbor in continuous_neighbors:
            umap_instance = emb_generator.fit_fuzzy_umap(
                df = df, 
                avail_contins=clustering_continuous, 
                avail_cats=clustering_categoricals, 
                continuous_neighbors = contin_neighbor, 
                categorical_neighbors = cat_neighbor,
                continuous_metric= 'euclidean',
                categorical_metric= 'jaccard',
                operator = '+'
            )
            i+=1
            print(f'Finished {i}')