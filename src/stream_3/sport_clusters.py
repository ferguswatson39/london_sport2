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

SPORTS_DESCRIPTIONS = {
    -1: "Noise",
    0: "Inactive",
    1: "Running + Walking (1)",
    2: "Yoga + Pilates",
    3: "Gymnastics",
    4: "Hyper Active - Athletics",
    5: "Dance + Golf",
    6: "Football + Tennis",
    7: "Hyper Active - Racket Sports",
    8: "Running + Walking (2)",
}

LCA_DESCRIPTIONS = {
    0.0: "Mid-life working fathers",
    1.0: "Young highly educated professional women",
    2.0: "Older unemployed disadvantaged adults",
    3.0: "Young adults living with parents",
    4.0: "Affluent older working women",
    5.0: "Oldest retired adults",
    6.0: "Mid-life professional women",
    7.0: "Highly educated professional fathers",
    8.0: "Young highly educated professionals",
    9.0: "Disabled mid-life adults",
    10.0: "Affluent older professionals",
    11.0: "Older middle socioeconomic workers",
    12.0: "Single mothers, lower socioeconomic group",
    13.0: "Oldest retired adults (Group B)",
    14.0: "Established professional adults",
    15.0: "Older transitioning retirees",
    16.0: "Young students living with parents",
    17.0: "University students in shared housing",
    18.0: "Young Asian professionals",
    19.0: "Lower socioeconomic family mothers",
    20.0: "Highly educated young professionals",
    21.0: "Professional mothers",
    22.0: "Working family mothers",
    23.0: "Retired older adults",
    24.0: "Affluent retired older adults",
}

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

def merge_clustering_data(sports_matrix_labeled : pd.DataFrame, serials, target_year = '2022/23'):
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_master_clustering_data_set.csv')
    clustering_df = pd.read_csv(df_path)
    clustering_df = clustering_df[clustering_df['year'] == target_year].dropna(subset = ['LCA_Class'])
    sports_matrix_labeled['serial'] = serials
    merged_df = pd.merge(sports_matrix_labeled[['serial', 'LABEL']], clustering_df[['serial', 'LCA_Class']], on = 'serial', how = 'inner')
    return merged_df

def plot_co_occurence(merged_df : pd.DataFrame):
    crosstab = pd.crosstab(merged_df['LABEL'], merged_df['LCA_Class'], normalize = 'index')
    baseline = merged_df['LCA_Class'].value_counts(normalize = True)
    lift_matrix = crosstab.div(baseline, axis = 1) - 1
    lift_matrix = lift_matrix.rename(index = SPORTS_DESCRIPTIONS, columns = LCA_DESCRIPTIONS)
    plt.figure(figsize = (20, 10))
    sns.heatmap(data = lift_matrix, annot = True, center = 0.0, cmap = "vlag", fmt = "+.1f", vmin = -1.0,  linecolor = 'white')
    plt.xticks(fontweight = 'bold', rotation = 90, ha = 'right')
    plt.yticks(fontweight = 'bold')
    plt.ylabel('Sporting Cluster', fontweight = 'bold', fontsize = 12)
    plt.xlabel('LCA Cluster', fontweight = 'bold', fontsize = 12)
    plt.tight_layout()
    plt.savefig(f"figures/co-occurence.png", bbox_inches = 'tight', dpi = 500)
    plt.show()

if __name__ == "__main__":
    sports_matrix = get_sports_matrix()
    sports_matrix = fix_hillwalk_duplication(sports_matrix)
    serials = sports_matrix['serial']
    sports_matrix = sports_matrix[SPORTS_NO_OVERLAP]
    generator = GenerateUmapEmb(sports_matrix)
    metric = 'cosine'
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
    merged_df = merge_clustering_data(sports_matrix_labeled, serials)
    plot_co_occurence(merged_df)