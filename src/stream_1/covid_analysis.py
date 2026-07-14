from pathlib import Path
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_data
import seaborn as sns
import matplotlib.pyplot as plt

MONTH_MAP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9 : 3, 10: 4, 11: 4, 12: 4}
QUARTER_MAP = {1: 2, 2: 5, 3: 8, 4: 11}
ADJUST_COVID = True

def prepare_monthly_data(ADJUST_COVID = False):
    monthly_df = get_data()
    monthly_df = correct_year_and_month(monthly_df)
    if ADJUST_COVID: monthly_df = covid_adjustment(monthly_df)
    monthly_df = monthly_df.groupby(['year', 'month'])['MEMS7_ALL'].agg(['mean', 'count']).reset_index()
    monthly_df['date'] = pd.to_datetime(monthly_df['year'].astype(str) + monthly_df['month'].astype(int).astype(str), format='%Y%m')
    return monthly_df

def prepare_quarterly_data(ADJUST_COVID = False):
    quarterly_df = get_data()
    quarterly_df = correct_year_and_month(quarterly_df)
    quarterly_df['quarter_num'] = quarterly_df['month'].map(MONTH_MAP)
    quarterly_df['quarter_date'] = pd.to_datetime(quarterly_df['year'].astype(str) + quarterly_df['quarter_num'].map(QUARTER_MAP).astype(str), format='%Y%m')
    quarterly_df = quarterly_df[~((quarterly_df['year'] == 2023) & (quarterly_df['quarter_num'] == 4))]
    if ADJUST_COVID: quarterly_df = covid_adjustment(quarterly_df)
    quarterly_df = quarterly_df.groupby(['year', 'quarter_date', 'quarter_num'])['MEMS7_ALL'].agg(['mean', 'count']).reset_index()
    return quarterly_df

def correct_year_and_month(dataframe : pd.DataFrame):
    dataframe['month'] = ((dataframe['month'] - 3) % 12) + 1
    dataframe['year'] = dataframe['year'].str.split('/').str[1].astype(int) + 2000
    dataframe.loc[dataframe['month'].isin([11,12]), 'year'] = dataframe['year'] - 1
    return dataframe

def covid_adjustment(dataframe : pd.DataFrame, target_col = 'MEMS7_ALL'):
    PRE_COVID = dataframe[dataframe['year'] < 2020]
    seasonal_average = PRE_COVID.groupby('month')[target_col].mean().reset_index()
    seasonal_average = seasonal_average.rename(columns = {target_col: 'seasonal_average'})
    dataframe = dataframe.merge(seasonal_average, on = 'month', how = 'left')
    IS_COVID = (((dataframe['year'] == 2020) & (dataframe['month'].isin([3, 4, 5, 11, 12]))) |
                ((dataframe['year'] == 2021) & (dataframe['month'].isin([1, 2, 3]))))
    dataframe.loc[IS_COVID, target_col] = dataframe.loc[IS_COVID, 'seasonal_average']
    return dataframe.drop(columns = ['seasonal_average'])

if __name__ == "__main__":
    # Monthly
    monthly_df = prepare_monthly_data(ADJUST_COVID = ADJUST_COVID)
    fig, ax = plt.subplots(figsize = (10, 4))
    sns.lineplot(data = monthly_df, x = 'date', y = 'mean', ax = ax, zorder = 1, color = 'grey')
    sns.scatterplot(data = monthly_df, x = 'date', y = 'mean', hue = 'year', ax = ax, size = 'count', sizes = (20, 100), legend = False, zorder = 2, palette = 'RdYlGn')
    ax.set_ylim(450, None)
    ax.set(xlabel = 'Year', ylabel = 'Average Participation')
    ax.axvspan('2020-03-01', '2021-01-01', alpha = 0.2, color = 'skyblue')
    ax.text(x = pd.to_datetime('2020-08-01'), y = 1000, s = 'COVID Lockdowns', color = 'skyblue', ha = 'center', fontweight = 'bold')
    if ADJUST_COVID: 
        plt.savefig('src/stream_1/figures/Monthly Participation (COVID Adjusted)', bbox_inches = 'tight')
    else:
        plt.savefig('src/stream_1/figures/Monthly Participation', bbox_inches = 'tight')

    # Quarterly
    quarterly_df = prepare_quarterly_data(ADJUST_COVID = ADJUST_COVID)
    fig, ax = plt.subplots(figsize = (10, 4))
    sns.lineplot(data = quarterly_df, x = 'quarter_date', y = 'mean', ax = ax, zorder = 1, color = 'grey')
    sns.scatterplot(data = quarterly_df, x = 'quarter_date', y = 'mean', hue = 'year', ax = ax, size = 'count', sizes = (60, 100), legend = False, zorder = 2, palette = 'RdYlGn')
    ax.set_ylim(450, None)
    ax.set(xlabel = 'Year', ylabel = 'Average Participation')
    ax.axvspan('2020-03-01', '2021-01-01', alpha = 0.2, color = 'skyblue')
    ax.text(x = pd.to_datetime('2020-08-01'), y = 950, s = 'COVID Lockdowns', color = 'skyblue', ha = 'center', fontweight = 'bold')
    if ADJUST_COVID: 
        plt.savefig('src/stream_1/figures/Quarterly Participation (COVID Adjusted)', bbox_inches = 'tight')
    else:
        plt.savefig('src/stream_1/figures/Quarterly Participation', bbox_inches = 'tight')