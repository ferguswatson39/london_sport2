from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_modality_data, get_sporting_distributions, get_monthly_data, get_quarterly_data
from src.stream_1.prepare_boroughs import prepare_borough_data, prepare_national_data
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
        plt.savefig('figures/profiling/cluster-geography', bbox_inches = 'tight', dpi = 500)
        plt.show()
    
    def plot_sport_distribution(self, dataframe):
        plt.figure(figsize = (8, 2.5))
        sns.barplot(dataframe, x = 'MEMS Contribution %', y = 'Sport', hue = 'Category')
        plt.legend('', frameon = False)
        plt.tight_layout()
        plt.ylabel(None)
        plt.savefig('figures/profiling/specific-sports', bbox_inches = 'tight', dpi = 500)
        plt.show()

    def plot_monthly(self, dataframe, ADJUST_COVID):
        _, ax = plt.subplots(figsize = (10, 2.25))
        sns.lineplot(data = dataframe, x = 'date', y = 'mean', ax = ax, zorder = 1, color = 'grey')
        sns.scatterplot(data = dataframe, x = 'date', y = 'mean', hue = 'year', ax = ax, size = 'count', sizes = (20, 100), legend = False, zorder = 2, palette = 'RdYlGn')
        ax.set_ylim(450, None)
        ax.set(ylabel = 'Average Participation')
        ax.xaxis.label.set_visible(False)
        ax.axvspan('2020-03-01', '2021-01-01', alpha = 0.2, color = 'skyblue')
        ax.text(x = pd.to_datetime('2020-08-01'), y = 850, s = 'COVID Lockdowns', color = 'skyblue', ha = 'center', fontweight = 'bold')
        if ADJUST_COVID: 
            plt.savefig('figures/seasonality/Monthly Participation (COVID Adjusted)', bbox_inches = 'tight')
        else:
            plt.savefig('figures/seasonality/Monthly Participation', bbox_inches = 'tight', dpi = 500)
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
            plt.savefig('figures/seasonality/Quarterly Participation (COVID Adjusted)', bbox_inches = 'tight')
        else:
            plt.savefig('figures/seasonality/figures/Quarterly Participation', bbox_inches = 'tight')
        plt.show()

    def plot_national_comparison(self, borough_df, national):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 6), sharey = True)
        for borough in borough_df['LA_Name'].unique():
            subset = borough_df[borough_df['LA_Name'] == borough].reset_index(drop = True)

            # Above Average Always
            if (subset['MEMS7_ALL'].values > national['MEMS7_ALL'].values).all():
                ax1.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.55, color = '#82B366', zorder = 2, linewidth = 1.5, marker='o', markersize = 4)
                ax1.plot(national['year'], national['MEMS7_ALL'], linestyle = 'dashed', color = 'black', linewidth = 3, zorder = 5)
                ax1.set_xticks(national['year'])
                ax1.set_xticklabels(national['year'], rotation = 45, ha = 'right')
                ax1.set_title('Always Above Average', weight = 'bold', fontsize = 15, color = '#82B366')
                if borough == 'Lambeth':
                    ax1.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -6), color = '#82B366',  fontweight = 'semibold', fontsize = 12, ha = 'left')
                elif borough == 'Westminster':
                    ax1.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -4), color = '#82B366',  fontweight = 'semibold', fontsize = 12, ha = 'left')
                else:
                    ax1.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, 0), color = '#82B366',  fontweight = 'semibold', fontsize = 12, ha = 'left')
                ax1.set_xlim(2017, 2024)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)


            # Below Average Always
            elif (subset['MEMS7_ALL'].values < national['MEMS7_ALL'].values).all():
                ax2.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.55, color = '#B85450', zorder = 2, linewidth = 1.5, marker='o', markersize = 4)
                ax2.plot(national['year'], national['MEMS7_ALL'], linestyle = 'dashed', color = 'black', linewidth = 3, zorder = 5)
                ax2.set_xticks(national['year'])
                ax2.set_xticklabels(national['year'], rotation = 45, ha = 'right')
                ax2.set_title('Always Below Average', weight = 'bold', fontsize = 15, color = '#B85450')
                if borough == 'Barking':
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -25), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' )  
                elif borough == 'Hillingdon':  
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -26), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' )  
                elif borough == 'Havering':  
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -20), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' )     
                elif borough == 'Croydon':  
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -20), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' ) 
                elif borough == 'Redbridge':  
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -10), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' )   
                elif borough == 'Bexley':  
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, -3), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' )  
                else:
                    ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (5, 0), color = '#B85450', fontweight = 'semibold', fontsize = 12, ha = 'left' )     
                ax2.set_xlim(2017, 2024)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)

            # Fluctuating
            else:
                ax3.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.55, color = '#777777', zorder = 2, linewidth = 1.5, marker='o', markersize = 4)
                ax3.plot(national['year'], national['MEMS7_ALL'], linestyle = 'dashed', color = 'black', linewidth = 3, zorder = 5)
                ax3.set_xticks(national['year'])
                ax3.set_xticklabels(national['year'], rotation = 45, ha = 'right')
                ax3.set_title('Fluctuating', weight = 'bold', fontsize = 15, color = "#777777")
                ax3.set_xlim(2017, 2024)
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)

        plt.savefig('figures/borough/National Comparison', bbox_inches = 'tight', dpi = 400)
        plt.show()  


if __name__ == "__main__":
    ADJUST_COVID = False
    modal_df = get_modality_data()
    distribution_df = get_sporting_distributions()
    monthly_df = get_monthly_data(ADJUST_COVID = ADJUST_COVID)
    quarterly_df = get_quarterly_data(ADJUST_COVID = ADJUST_COVID)
    borough_df = prepare_borough_data()
    national = prepare_national_data()
    borough_df['LA_Name'] = borough_df['LA_Name'].replace({'Barking and Dagenham' : 'Barking', 'Hammersmith and Fulham' : 'Hammersmith'})
    borough_df = borough_df[borough_df['year'] != 2016]
    national = national[national['year'] != 2016]
    analysis = AnalysisPlots()
    analysis.plot_modality_graph(modal_df)
    analysis.plot_quarterly(quarterly_df, ADJUST_COVID = ADJUST_COVID)
    analysis.plot_national_comparison(borough_df, national)
    analysis.plot_monthly(monthly_df, ADJUST_COVID = ADJUST_COVID)
    analysis.plot_sport_distribution(distribution_df)
