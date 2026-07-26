from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_sports_matrix
from src.stream_3.gen_umap_emb import GenerateUmapEmb
from src.stream_3.gen_hdb_clus import GenerateHDBSCAN
import seaborn as sns
import matplotlib.pyplot as plt

SPORTS_NO_OVERLAP = ['WALKALL', 'RUNNING', 'ATHLETICS', 'HILLWALK', 'CYCALL', # Walking, Running, Cycling
                     'GYM', 'EXMACHINES', 'WEIGHTS', 'BODYWEIGHT', # Strength and Conditioning
                     'FITCLASS', 'DANCECLASS', 'YOGA', 'PILATES', # Exercise Classes
                     'FOOTBALL', 'CRICKET', 'BASKETBALL', 'RUGBYUNION', 'NETBALL', # Team Sports
                     'TENNIS', 'BADMINTON', 'TABLETENNIS', 'SQUASH', # Racket Sports
                     'BOXING', 'MARTIAL', # Combat Sports
                     'SWIM', 'WATERSPORTS', # Water Sports - including Rowing
                     'GOLF', 'EQUEST', 'GYMNASTICS', 'CLIMBBOULD',  'ACTPLAY', 'SNOWSPORT'] # Other

def plot_sports_clusters(dataframe : pd.DataFrame, metric : str):
    plt.figure(figsize = (12, 8))
    sns.scatterplot(data = dataframe, x = 'DIM1', y = 'DIM2', hue = 'LABEL', s = 150, palette = 'tab10', alpha = 0.8)
    plt.tight_layout()
    plt.savefig(f"figures/sports-clusters-{metric}", bbox_inches = 'tight', dpi = 500)
    plt.show()

sports_matrix, cols = get_sports_matrix()
sports_matrix = sports_matrix[SPORTS_NO_OVERLAP]
generator = GenerateUmapEmb(sports_matrix)
METRICS = ['cosine', 'jaccard']
for metric in METRICS:
    umap = generator.fit_umap_sports(metric = metric)
    coordinates = umap.embedding_
    hdbscan = GenerateHDBSCAN(coordinates)
    labels = hdbscan.hdb_narrowed().labels_
    plot_df = pd.DataFrame({'DIM1': coordinates[:, 0], 'DIM2': coordinates[:, 1], 'LABEL': labels})
    plot_sports_clusters(plot_df, metric = metric)

