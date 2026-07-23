from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_modality_data, get_sporting_distributions, get_monthly_data, get_quarterly_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AnalysisPlots:

    def annotate_modality_plot(self, ax, dataframe):
        colors = ax.collections[0].get_facecolors()
        for i, row in dataframe.iterrows():
            ax.annotate(f"{int(row['Percentage'])}%", 
                        (row['LCA_Class'], row['LA']), 
                        xytext = (0, 6), 
                        textcoords = 'offset points', 
                        ha = 'center', 
                        color = colors[i])
        
    def plot_modality_graph(self, dataframe):
        plt.figure(figsize = (9, 4.5))
        ax = sns.scatterplot(data = dataframe, x = 'LCA_Class', y = 'LA', hue = 'LA', size = 'Percentage', sizes = (75, 150), zorder = 3)
        plt.legend('', frameon = False)
        plt.ylabel('Modal Borough', fontweight = 'bold', fontsize = 12)
        plt.xlabel('Cluster', fontweight = 'bold', fontsize = 12)
        plt.xticks(dataframe['LCA_Class'])
        plt.grid(visible = True, linestyle = '--', alpha = 0.2, zorder = -1)
        plt.tight_layout()
        AnalysisPlots.annotate_modality_plot(self, ax, dataframe)
        min, max = ax.get_ylim()
        ax.set_ylim(min, max - 0.3)
        plt.savefig('figures/cluster-geography', bbox_inches = 'tight', dpi = 500)
        plt.show()
    
    def plot_sport_distribution(self, dataframe):
        sns.barplot(dataframe, x = 'MEMS Contribution %', y = 'Sport', hue = 'Category')
        plt.legend('', frameon = False)
        plt.tight_layout()
        plt.ylabel(None)
        plt.savefig('figures/specific-sports', bbox_inches = 'tight', dpi = 500)
        plt.show()

    def plot_monthly(self, dataframe, ADJUST_COVID):
        _, ax = plt.subplots(figsize = (10, 4))
        sns.lineplot(data = dataframe, x = 'date', y = 'mean', ax = ax, zorder = 1, color = 'grey')
        sns.scatterplot(data = dataframe, x = 'date', y = 'mean', hue = 'year', ax = ax, size = 'count', sizes = (20, 100), legend = False, zorder = 2, palette = 'RdYlGn')
        ax.set_ylim(450, None)
        ax.set(xlabel = 'Year', ylabel = 'Average Participation')
        ax.axvspan('2020-03-01', '2021-01-01', alpha = 0.2, color = 'skyblue')
        ax.text(x = pd.to_datetime('2020-08-01'), y = 975, s = 'COVID Lockdowns', color = 'skyblue', ha = 'center', fontweight = 'bold')
        if ADJUST_COVID: 
            plt.savefig('src/stream_1/figures/Monthly Participation (COVID Adjusted)', bbox_inches = 'tight')
        else:
            plt.savefig('src/stream_1/figures/Monthly Participation', bbox_inches = 'tight')
        plt.show()
    
    def plot_quarterly(self, dataframe, ADJUST_COVID):
        _, ax = plt.subplots(figsize = (10, 4))
        sns.lineplot(data = dataframe, x = 'quarter_date', y = 'mean', ax = ax, zorder = 1, color = 'grey')
        sns.scatterplot(data = dataframe, x = 'quarter_date', y = 'mean', hue = 'year', ax = ax, size = 'count', sizes = (60, 100), legend = False, zorder = 2, palette = 'RdYlGn')
        ax.set_ylim(450, None)
        ax.set(xlabel = 'Year', ylabel = 'Average Participation')
        ax.axvspan('2020-03-01', '2021-01-01', alpha = 0.2, color = 'skyblue')
        ax.text(x = pd.to_datetime('2020-08-01'), y = 900, s = 'COVID Lockdowns', color = 'skyblue', ha = 'center', fontweight = 'bold')
        if ADJUST_COVID: 
            plt.savefig('src/stream_1/figures/Quarterly Participation (COVID Adjusted)', bbox_inches = 'tight')
        else:
            plt.savefig('src/stream_1/figures/Quarterly Participation', bbox_inches = 'tight')
        plt.show()


if __name__ == "__main__":
    ADJUST_COVID = False
    modal_df = get_modality_data()
    distribution_df = get_sporting_distributions()
    monthly_df = get_monthly_data(ADJUST_COVID = ADJUST_COVID)
    quarterly_df = get_quarterly_data(ADJUST_COVID = ADJUST_COVID)
    analysis = AnalysisPlots()
    analysis.plot_modality_graph(modal_df)
    analysis.plot_sport_distribution(distribution_df)
    analysis.plot_monthly(monthly_df, ADJUST_COVID = ADJUST_COVID)
    analysis.plot_quarterly(quarterly_df, ADJUST_COVID = ADJUST_COVID)
