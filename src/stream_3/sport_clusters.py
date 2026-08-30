from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_sports_matrix
from stream_3.umap.gen_umap_emb import GenerateUmapEmb
from stream_3.hdb.gen_hdb_clus import GenerateHDBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

SPORTS_NO_OVERLAP = ['WALKALL', 'RUNNING', 'ATHLETICS', 'HILLWALK', 'CYCALL', # Walking, Running, Cycling
                     'GYM', 'EXMACHINES', 'WEIGHTS', 'BODYWEIGHT', # Strength and Conditioning
                     'FITCLASS', 'DANCECLASS', 'YOGA', 'PILATES', # Exercise Classes
                     'FOOTBALL', 'CRICKET', 'BASKETBALL', 'RUGBYUNION', 'NETBALL', # Team Sports
                     'TENNIS', 'BADMINTON', 'TABLETENNIS', 'SQUASH', # Racket Sports
                     'BOXING', 'MARTIAL', # Combat Sports
                     'SWIM', 'WATERSPORTS', # Water Sports - including Rowing
                     'GOLF', 'EQUEST', 'GYMNASTICS', 'CLIMBBOULD',  'ACTPLAY', 'SNOWSPORT'] # Other

SPORTS_DESCRIPTIONS = { -1: "Noise",
                        0: "Inactive",
                        1: "Running + Walking (1)",
                        2: "Yoga + Pilates",
                        3: "Gymnastics",
                        4: "Hyper Active - Athletics",
                        5: "Dance + Golf",
                        6: "Football + Tennis",
                        7: "Hyper Active - Racket Sports",
                        8: "Running + Walking (2)"}

LCA_DESCRIPTIONS = {
    0.0:  "Middle-aged Working Fathers",
    1.0:  "Highly Educated Working Mothers",
    2.0:  "Professional Fathers",
    3.0:  "Professional Part-time Working Mothers",
    4.0:  "Later-career Middle Socio-economic Workers",
    5.0:  "Highly Educated Early Retirees",
    6.0:  "Black Middle-aged Professional Women",
    7.0:  "Established White British Retirees",
    8.0:  "Young Students in Shared Housing",
    9.0:  "Young and Mid-career Middle Socio-economic Workers",
    10.0: "Highly Motivated Young Professional Men",
    11.0: "Young Graduate Professionals in Shared Housing",
    12.0: "Later-career Professional Women",
    13.0: "Unemployed Lower Socio-economic Adults",
    14.0: "Long-term Sick and Disabled Adults",
    15.0: "Midlife Professional Workers Living Alone",
    16.0: "Lower Socio-economic Mothers and Carers",
    17.0: "Young Professional Women",
    18.0: "Older Parents Approaching Retirement",
    19.0: "Independent Older Retirees",
    20.0: "Highly Educated White Other Professionals",
    21.0: "Young Students Living with Parents",
    22.0: "Older Retirees with Lower Qualifications",
    23.0: "Later-career Professional Men",
    24.0: "Disabled Older Retirees",
    25.0: "Young Lower Socio-economic Adults Living with Parents",
    26.0: "Young Professionals Living with Parents"
}

class SportsClustering:
    def __init__(self, metric : str = 'cosine', target_year : str = '2022/23'):
        self.metric = metric
        self.target_year = target_year
        self.serials = None
        self.sports_matrix = None
        self.sports_matrix_labeled = None

    def fix_hillwalk_duplication(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        hillwalk = dataframe['HILLWALK'].max(axis = 1)
        dataframe = dataframe.drop(columns = ['HILLWALK'])
        dataframe['HILLWALK'] = hillwalk
        return dataframe

    def plot_sports_clusters(self, dataframe : pd.DataFrame):
        plt.figure(figsize = (12, 8))
        sns.scatterplot(data = dataframe, x = 'DIM1', y = 'DIM2', hue = 'LABEL', s = 150, palette = 'tab10', alpha = 0.8)
        plt.tight_layout()
        plt.savefig(f"figures/profiling/sports-clusters-{self.metric}", bbox_inches = 'tight', dpi = 500)
        plt.show()

    def calculate_lift_matrix(self, dataframe : pd.DataFrame) -> pd.DataFrame:
        baseline = dataframe.drop(columns = ['LABEL']).mean(axis = 0)
        clusters = dataframe.groupby('LABEL').mean() 
        lift_matrix = clusters.div(baseline, axis = 1) - 1
        return lift_matrix

    def plot_lift_matrix(self, dataframe : pd.DataFrame):
        plt.figure(figsize = (20, 10))
        sns.heatmap(data = dataframe, annot = True, center = 0.0, cmap = "vlag", fmt = "+.1f", vmin = -1.0, vmax = 3.0, linecolor = 'white')
        plt.tight_layout()
        plt.xticks(fontweight = 'bold')
        plt.ylabel('Sporting Cluster', fontweight = 'bold', fontsize = 12)
        plt.savefig(f"figures/profiling/lift-matrix-{self.metric}", bbox_inches = 'tight', dpi = 500)
        plt.show()

    def merge_clustering_data(self) -> pd.DataFrame:
        df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_clustering_output_data.csv')
        clustering_df = pd.read_csv(df_path)
        clustering_df = clustering_df[clustering_df['year'] == self.target_year].dropna(subset = ['LCA_Class'])
        self.sports_matrix_labeled['serial'] = self.serials
        merged_df = pd.merge(self.sports_matrix_labeled[['serial', 'LABEL']], clustering_df[['serial', 'LCA_Class']], on = 'serial', how = 'inner')
        return merged_df

    def plot_co_occurence(self, merged_df : pd.DataFrame):
        crosstab = pd.crosstab(merged_df['LABEL'], merged_df['LCA_Class'], normalize = 'index')
        baseline = merged_df['LCA_Class'].value_counts(normalize = True)
        lift_matrix = crosstab.div(baseline, axis = 1) - 1
        lift_matrix = lift_matrix.rename(index = SPORTS_DESCRIPTIONS, columns = LCA_DESCRIPTIONS)
        plt.figure(figsize = (20, 10))
        sns.heatmap(data = lift_matrix, annot = True, center = 0.0, cmap = "vlag", fmt = "+.1f", vmin = -1.0,  linecolor = 'white')
        plt.xticks(fontweight = 'bold', rotation = 45, ha = 'right')
        plt.yticks(fontweight = 'bold')
        plt.ylabel('Sporting Preference', fontweight = 'bold', fontsize = 15)
        plt.xlabel('Population Sub-Group', fontweight = 'bold', fontsize = 15)
        plt.tight_layout()
        plt.savefig(f"figures/profiling/co-occurence.png", bbox_inches = 'tight', dpi = 500)
        plt.show()

    def prepare_persona_proportions(self) -> pd.DataFrame:
        df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_clustering_output_data.csv')
        clustering_df = pd.read_csv(df_path)
        clustering_df = clustering_df[clustering_df['year'] == self.target_year].dropna(subset = ['LCA_Class'])
        self.sports_matrix_labeled['serial'] = self.serials
        merged_df = pd.merge(self.sports_matrix_labeled[['serial', 'LABEL']], clustering_df[['serial', 'motivd_POP', 'NSSEC5', 'Age9', 'Disab2_POP', 'IMD10']], on = 'serial', how = 'inner')
        merged_df['Deprived %'] = merged_df['IMD10'].isin([1, 2, 3]).astype(int)
        merged_df['Older %'] = merged_df['Age9'].isin([7, 8, 9]).astype(int)
        merged_df['Affluent %'] = merged_df['NSSEC5'].isin([1, 2]).astype(int)
        merged_df['Extrinsically Motivated %'] = merged_df['motivd_POP'].isin([1, 2]).astype(int)
        merged_df['Disabled %'] = merged_df['Disab2_POP'].isin([1]).astype(int)
        cols = ['Deprived %', 'Older %', 'Affluent %', 'Extrinsically Motivated %', 'Disabled %']
        proportions = merged_df.groupby('LABEL')[cols].mean() * 100
        proportions = proportions.div(proportions.max(axis = 0), axis = 1) * 100
        return proportions.reset_index()

    def plot_radar(self, dataframe: pd.DataFrame, sports : list = [0, 2, 4, 6]):
        dataframe['Sport'] = dataframe['LABEL'].map(SPORTS_DESCRIPTIONS)
        dataframe = dataframe[dataframe['LABEL'].isin(sports)].drop(columns = ['LABEL'])
        dataframe = dataframe.melt(id_vars = ['Sport'], var_name = 'Trait', value_name = 'Percentage')
        fig = px.line_polar(data_frame = dataframe, r = 'Percentage', theta = 'Trait', color = 'Sport', markers = True, line_close = True, template = 'simple_white')
        fig.update_layout(polar = dict(radialaxis = dict(visible = False)))
        fig.write_image("figures/radar.png")
        fig.show()

if __name__ == "__main__":
    sc = SportsClustering()
    sports_matrix = get_sports_matrix()
    sports_matrix = sc.fix_hillwalk_duplication(sports_matrix)
    sc.serials = sports_matrix['serial']
    sc.sports_matrix = sports_matrix[SPORTS_NO_OVERLAP]
    generator = GenerateUmapEmb(sc.sports_matrix)
    metric = 'cosine'
    umap = generator.fit_umap_sports(metric = sc.metric)
    coordinates = umap.embedding_
    hdbscan = GenerateHDBSCAN(coordinates)
    labels = hdbscan.hdb_narrowed().labels_
    plot_df = pd.DataFrame({'DIM1': coordinates[:, 0], 'DIM2': coordinates[:, 1], 'LABEL': labels})
    sc.plot_sports_clusters(plot_df)
    sc.sports_matrix_labeled = sc.sports_matrix.copy()
    sc.sports_matrix_labeled['LABEL'] = labels
    lift_matrix = sc.calculate_lift_matrix(sc.sports_matrix_labeled)
    sc.plot_lift_matrix(lift_matrix)
    merged_df = sc.merge_clustering_data()
    sc.plot_co_occurence(merged_df)
    proportions = sc.prepare_persona_proportions()
    sc.plot_radar(proportions)