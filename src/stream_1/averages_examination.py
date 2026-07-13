from prepare_boroughs import prepare_borough_data, prepare_national_data
import seaborn as sns
import matplotlib.pyplot as plt

# Load Borough and National Data
borough_df = prepare_borough_data()
borough_df['LA_Name'] = borough_df['LA_Name'].replace({'Kingston upon Thames': 'Kingston', 'Richmond upon Thames': 'Richmond', 'Barking and Dagenham' : 'Barking', 'Hammersmith and Fulham' : 'Hammersmith'})
national = prepare_national_data()

# Define COVID constants
PRE_COVID_YEAR = '2018/19'
COVID_YEAR = '2019/20'
POST_COVID_YEAR1 = '2020/21'
POST_COVID_YEAR2 = '2021/22'
POST_COVID_YEAR3 = '2022/23'

def plot_national_comparison():
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (25, 6), sharey = True)
    for borough in borough_df['LA_Name'].unique():
        subset = borough_df[borough_df['LA_Name'] == borough].reset_index(drop = True)

        # Above Average Always
        if (subset['MEMS7_ALL'] > national['MEMS7_ALL']).all():
            ax1.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.55, color = '#82B366', zorder = 2, linewidth = 1.5, marker='o', markersize = 4)
            ax1.plot(national['year'], national['MEMS7_ALL'], linestyle = 'dashed', color = 'black', linewidth = 3, zorder = 5)
            ax1.set_xticklabels(national['year'], rotation = 45, ha = 'right')
            ax1.set_title('Always Above Average', weight = 'bold', fontsize = 15, color = '#82B366')
            if borough == 'Lambeth':
                ax1.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (12, -5), color = '#82B366',  fontweight = 'semibold', fontsize = 9)
            else:
                ax1.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (12, 0), color = '#82B366',  fontweight = 'semibold', fontsize = 9)
            ax1.set_aspect(0.023) 

        # Below Average Always
        elif (subset['MEMS7_ALL'] < national['MEMS7_ALL']).all():
            ax2.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.55, color = '#B85450', zorder = 2, linewidth = 1.5, marker='o', markersize = 4)
            ax2.plot(national['year'], national['MEMS7_ALL'], linestyle = 'dashed', color = 'black', linewidth = 3, zorder = 5)
            ax2.set_xticklabels(national['year'], rotation = 45, ha = 'right')
            ax2.set_title('Always Below Average', weight = 'bold', fontsize = 15, color = '#B85450')
            if borough == 'Newham':
                ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (12, 1), color = '#B85450', fontweight = 'semibold', fontsize = 9) 
            else:
                ax2.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (12, -3), color = '#B85450', fontweight = 'semibold', fontsize = 9)     
            ax2.set_aspect(0.022) 

        # Fluctuating
        else:
            PRE_COVID_BOROUGH = subset.loc[subset['year'] == PRE_COVID_YEAR, 'MEMS7_ALL'].values
            PRE_COVID_NATIONAL = national.loc[national['year'] == PRE_COVID_YEAR, 'MEMS7_ALL'].values
            COVID_BOROUGH = subset.loc[subset['year'] == COVID_YEAR, 'MEMS7_ALL'].values
            COVID_NATIONAL = national.loc[national['year'] == COVID_YEAR, 'MEMS7_ALL'].values
            POST_COVID_BOROUGH1 = subset.loc[subset['year'] == POST_COVID_YEAR1, 'MEMS7_ALL'].values
            POST_COVID_NATIONAL1 = national.loc[national['year'] == POST_COVID_YEAR1, 'MEMS7_ALL'].values
            POST_COVID_BOROUGH2 = subset.loc[subset['year'] == POST_COVID_YEAR2, 'MEMS7_ALL'].values
            POST_COVID_NATIONAL2 = national.loc[national['year'] == POST_COVID_YEAR2, 'MEMS7_ALL'].values
            POST_COVID_BOROUGH3 = subset.loc[subset['year'] == POST_COVID_YEAR3, 'MEMS7_ALL'].values
            POST_COVID_NATIONAL3 = national.loc[national['year'] == POST_COVID_YEAR3, 'MEMS7_ALL'].values

            # Yet to recover from COVID
            if (PRE_COVID_BOROUGH > PRE_COVID_NATIONAL or COVID_BOROUGH > COVID_NATIONAL) and POST_COVID_BOROUGH1 < POST_COVID_NATIONAL1 and POST_COVID_BOROUGH2 < POST_COVID_NATIONAL2 and POST_COVID_BOROUGH3 < POST_COVID_NATIONAL3:
                ax3.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.7, color = "#844D17", zorder = 3, linewidth = 1.5, marker='o', markersize = 4)
                ax3.annotate(borough, xy = (subset.iloc[-1]['year'], subset.iloc[-1]['MEMS7_ALL']), textcoords = 'offset points', xytext = (10, 0), color = "#844D17", fontweight = 'semibold', fontsize = 9)  
            else:
                ax3.plot(subset['year'], subset['MEMS7_ALL'], alpha = 0.55, color = '#FFCC99', zorder = 2, linewidth = 1.5, marker='o', markersize = 4)
            ax3.plot(national['year'], national['MEMS7_ALL'], linestyle = 'dashed', color = 'black', linewidth = 3, zorder = 5)
            ax3.set_xticklabels(national['year'], rotation = 45, ha = 'right')
            ax3.set_aspect(0.022) 
            ax3.set_title('Fluctuating', weight = 'bold', fontsize = 15, color = "#EABA8A") 
    plt.savefig('src/stream_1//figures/National Comparison', bbox_inches = 'tight')
    plt.show()
plot_national_comparison()