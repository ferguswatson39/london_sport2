from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_modality_data
import matplotlib.pyplot as plt
import seaborn as sns

def annotate_modality_plot(ax, dataframe):
    colors = ax.collections[0].get_facecolors()
    for i, row in dataframe.iterrows():
        ax.annotate(f"{int(row['Percentage'])}%", 
                    (row['LCA_Class'], row['LA']), 
                    xytext = (0, 6), 
                    textcoords = 'offset points', 
                    ha = 'center', 
                    color = colors[i])
        
def plot_modality_graph(modal_df):
    plt.figure(figsize = (9, 4.5))
    ax = sns.scatterplot(data = modal_df, x = 'LCA_Class', y = 'LA', hue = 'LA', size = 'Percentage', sizes = (75, 150), zorder = 3)
    plt.legend('', frameon = False)
    plt.ylabel('Modal Borough', fontweight = 'bold', fontsize = 12)
    plt.xlabel('Cluster', fontweight = 'bold', fontsize = 12)
    plt.xticks(modal_df['LCA_Class'])
    plt.grid(visible = True, linestyle = '--', alpha = 0.2, zorder = -1)
    plt.tight_layout()
    annotate_modality_plot(ax, modal_df)
    min, max = ax.get_ylim()
    ax.set_ylim(min, max - 0.3)
    plt.savefig('figures/cluster-geography', bbox_inches = 'tight', dpi = 500)
    plt.show()

if __name__ == "__main__":
    modal_df = get_modality_data()
    plot_modality_graph(modal_df)
