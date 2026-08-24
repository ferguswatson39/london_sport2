import pickle
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))

cluster_output_path = ROOT / 'data' / 'master_data' / '2016_to_2023_clustering_output_data.csv'
hdb_dict_path = ROOT / 'src' / 'stream_3'/ 'hdb_clusters' / 'hdb_cluster_dict.pkl'

print('Retrieving hdbscan labels...')

with open(hdb_dict_path, 'rb') as file:
    label_dict = pickle.load(file)

hdb_labels = label_dict['EMB_UMAP_conn_281_catn_50_conm_euclidean_catm_jaccard_plus.npy']['labels']

df = pd.read_csv(cluster_output_path)

df['hdb_labels_281_50'] = hdb_labels

print(f'Saving hdbscan labels to : {cluster_output_path}')
df.to_csv(cluster_output_path, index=False)