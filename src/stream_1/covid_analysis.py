from pathlib import Path
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_data
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_monthly_data():
    df = get_data()
    monthly_df = df.groupby(['year', 'month'])['MEMS7_ALL'].agg(['mean', 'count']).reset_index()
    monthly_df['month'] = ((monthly_df['month'] - 3) % 12) + 1 # Fix MonthTwelve 1 = November
    monthly_df['year'] = monthly_df['year'].str.split('/').str[1].astype(int) + 2000
    monthly_df.loc[monthly_df['month'].isin([11,12]), 'year'] = monthly_df['year'] - 1
    monthly_df['date'] = pd.to_datetime(monthly_df['year'].astype(str) + monthly_df['month'].astype(int).astype(str), format='%Y%m')
    return monthly_df

if __name__ == "__main__":
    monthly_df = prepare_monthly_data()
    fig, ax = plt.subplots(figsize = (10, 4))
    sns.lineplot(data = monthly_df, x = 'date', y = 'mean', ax = ax, zorder = 1, color = 'grey')
    sns.scatterplot(data = monthly_df, x = 'date', y = 'mean', hue = 'year', ax = ax, size = 'count', sizes = (20, 100), legend = False, zorder = 2, palette = 'RdYlGn')
    ax.set_ylim(450, None)
    ax.set(xlabel = 'Year', ylabel = 'Average Participation')
    ax.axvspan('2020-03-01', '2021-01-01', alpha = 0.2, color = 'skyblue')
    ax.text(x = pd.to_datetime('2020-08-01'), y = 1000, s = 'COVID Lockdowns', color = 'skyblue', ha = 'center', fontweight = 'bold')
    plt.savefig('src/stream_1/figures/Monthly Participation', bbox_inches = 'tight')
    plt.show()