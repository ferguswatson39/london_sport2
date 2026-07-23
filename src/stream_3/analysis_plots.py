from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_modality_data, get_sporting_distributions
import matplotlib.pyplot as plt
import seaborn as sns

class AnalysisPlots:

    def annotate_modality_plot(ax, dataframe):
        colors = ax.collections[0].get_facecolors()
        for i, row in dataframe.iterrows():
            ax.annotate(f"{int(row['Percentage'])}%", 
                        (row['LCA_Class'], row['LA']), 
                        xytext = (0, 6), 
                        textcoords = 'offset points', 
                        ha = 'center', 
                        color = colors[i])
        
    def plot_modality_graph(dataframe):
        plt.figure(figsize = (9, 4.5))
        ax = sns.scatterplot(data = dataframe, x = 'LCA_Class', y = 'LA', hue = 'LA', size = 'Percentage', sizes = (75, 150), zorder = 3)
        plt.legend('', frameon = False)
        plt.ylabel('Modal Borough', fontweight = 'bold', fontsize = 12)
        plt.xlabel('Cluster', fontweight = 'bold', fontsize = 12)
        plt.xticks(dataframe['LCA_Class'])
        plt.grid(visible = True, linestyle = '--', alpha = 0.2, zorder = -1)
        plt.tight_layout()
        AnalysisPlots.annotate_modality_plot(ax, dataframe)
        min, max = ax.get_ylim()
        ax.set_ylim(min, max - 0.3)
        plt.savefig('figures/cluster-geography', bbox_inches = 'tight', dpi = 500)
        plt.show()
    
    def plot_sport_distribution(dataframe):
        sns.barplot(dataframe, x = 'MEMS Contribution %', y = 'Sport', hue = 'Category')
        plt.legend('', frameon = False)
        plt.tight_layout()
        plt.ylabel(None)
        plt.savefig('figures/specific-sports', bbox_inches = 'tight', dpi = 500)
        plt.show()


if __name__ == "__main__":
    modal_df = get_modality_data()
    distribution_df = get_sporting_distributions()
    analysis = AnalysisPlots()
    analysis.plot_modality_graph(modal_df)
    analysis.plot_sport_distribution(distribution_df)
