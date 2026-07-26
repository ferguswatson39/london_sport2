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

def fix_hillwalk_duplication(dataframe: pd.DataFrame):
    hillwalk = dataframe['HILLWALK'].max(axis = 1)
    dataframe = dataframe.drop(columns = ['HILLWALK'])
    dataframe['HILLWALK'] = hillwalk
    return dataframe

def plot_sports_clusters(dataframe : pd.DataFrame, metric : str):
    plt.figure(figsize = (12, 8))
    sns.scatterplot(data = dataframe, x = 'DIM1', y = 'DIM2', hue = 'LABEL', s = 150, palette = 'tab10', alpha = 0.8)
    plt.tight_layout()
    plt.savefig(f"figures/sports-clusters-{metric}", bbox_inches = 'tight', dpi = 500)
    plt.show()

def calculate_lift_matrix(dataframe : pd.DataFrame):
    baseline = dataframe.drop(columns = ['LABEL']).mean(axis = 0)
    clusters = dataframe.groupby('LABEL').mean() 
    lift_matrix = clusters.div(baseline, axis = 1) - 1
    return lift_matrix

def plot_lift_matrix(dataframe : pd.DataFrame, metric : str):
    plt.figure(figsize = (20, 10))
    sns.heatmap(data = dataframe, annot = True, center = 0.0, cmap = "vlag", fmt = "+.1f", vmin = -1.0, vmax = 3.0, linecolor = 'white')
    plt.tight_layout()
    plt.xticks(fontweight = 'bold')
    plt.ylabel('Sporting Cluster', fontweight = 'bold', fontsize = 12)
    plt.savefig(f"figures/lift-matrix-{metric}", bbox_inches = 'tight', dpi = 500)
    plt.show()

sports_matrix, cols = get_sports_matrix()
sports_matrix = fix_hillwalk_duplication(sports_matrix)
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
    sports_matrix_labeled = sports_matrix.copy()
    sports_matrix_labeled['LABEL'] = labels
    lift_matrix = calculate_lift_matrix(sports_matrix_labeled)
    plot_lift_matrix(lift_matrix, metric = metric)
