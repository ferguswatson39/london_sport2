from pathlib import Path
import sys 
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import re
from natsort import natsorted

def retrieve_sort_emb_paths():
    embedding_folder = ROOT / 'src' / 'stream_3' / 'embeddings' 
    embedding_paths = [file for file in embedding_folder.iterdir() if file.name != 'classes']
    all_sorted = []
    for i in [500, 281, 158, 89, 50]:
        paths = [path for path in embedding_paths if re.search(fr'conn_{i}_', path.name)]
        paths = natsorted(paths)
        all_sorted += paths
    return all_sorted

if __name__ == '__main__':
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))
    axes = axes.flatten()
    all_sorted = retrieve_sort_emb_paths()

    for idx, path in enumerate(all_sorted):
        emb = np.load(path)
        axes[idx].scatter(emb[:,0], emb[:,1], s=0.05, alpha=0.1)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    fig.supxlabel('Categorical n_neighbors: 50 to 500 (left to right)')
    fig.supylabel('Continuous n_neighbors n_neighbors: 500 to 50 (top to bottom)')
    fig.savefig('full_emb_plot.png')



