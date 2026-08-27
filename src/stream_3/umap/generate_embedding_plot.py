from pathlib import Path
import sys 
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from natsort import natsorted
from stream_3.hdb.gen_hdb_clus import GenerateHDBSCAN2

def retrieve_sort_emb_paths():
    embedding_folder = ROOT / 'src' / 'stream_3' / 'embeddings' 
    embedding_paths = [file for file in embedding_folder.iterdir() if file.name != 'classes']
    all_sorted = []
    for i in [500, 281, 158, 89, 50]:
        f_name = f'conn_{i}_'
        paths = [path for path in embedding_paths if f_name in path.name]
        paths = natsorted(paths)
        all_sorted += paths
    return all_sorted

def make_n_cluster_dict(sorted_paths : list[Path], cluster_dict_path : Path) -> dict:
    print('Making cluster dictionary')
    cluster_dict = {}
    for path in tqdm(sorted_paths):
        emb = np.load(path)
        hdb = GenerateHDBSCAN2()
        hdb.fit(emb)
        labels = hdb.get_model().labels_
        dbcv = hdb.get_dbcv()
        # -1 to remove inclusion of noise grouping in n_custers
        n_unique_ints = np.unique(labels)
        if -1 in n_unique_ints:
            n_unique_clusters = len(n_unique_ints) -1 
        else:
            n_unique_clusters = len(n_unique_ints)
            
        cluster_dict[path.name] = {'n_clusters' : n_unique_clusters, 'labels' : labels, 'dbcv' : dbcv}

    print('Finished making cluster dictionary')
    print(f'Saving cluster dict to: {cluster_dict_path}')

    with open(cluster_dict_path, 'wb') as file:
        pickle.dump(cluster_dict, file)
    
    return cluster_dict

if __name__ == '__main__':
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()

    print('Retrieving embedding paths')
    sorted_paths = retrieve_sort_emb_paths()
    cluster_dict_path = ROOT / 'src' / 'stream_3' / 'hdb_clusters'  / 'hdb_cluster_dict.pkl'
    print('Checking if cluster dictionary has been found...')

    if not cluster_dict_path.exists():
        print('Cluster dictionary has not been found.')
        print('Proceeding to create cluster dictionary...')

        cluster_dict = make_n_cluster_dict(sorted_paths, cluster_dict_path)

        print('Finished cluster dictionary')
    else:
        print('Cluster dictionary has been location')
        print('Loading cluster dictinary...')

        with open(cluster_dict_path, 'rb') as file:
            cluster_dict = pickle.load(file)

    for idx, path in enumerate(sorted_paths):
        labels = cluster_dict[path.name]['labels']
        n_clusters = cluster_dict[path.name]['n_clusters']
        dbcv = cluster_dict[path.name]['dbcv']

        emb = np.load(path)

        axes[idx].scatter(emb[:,0], emb[:,1], s=0.05, alpha=0.1, c=labels)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_title(f'n = {n_clusters} | dbcv : {dbcv:.2f}', fontweight='bold')


    fig.supxlabel('Categorical n_neighbors: 50 to 500 (left to right)', fontsize=16, fontweight='bold', y=0.08)
    fig.supylabel('Continuous n_neighbors: 500 to 50 (top to bottom)', fontsize=16, fontweight='bold', x=0.08)

    save_path = ROOT / 'figures' / 'umap' / 'full_emb_plot.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=500)



