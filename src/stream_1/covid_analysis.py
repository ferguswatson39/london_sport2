from pathlib import Path
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

class Covid:
    def correct_year_and_month(self, dataframe : pd.DataFrame):
        dataframe['month'] = ((dataframe['month'] - 3) % 12) + 1
        dataframe['year'] = dataframe['year'].str.split('/').str[1].astype(int) + 2000
        dataframe.loc[dataframe['month'].isin([11,12]), 'year'] = dataframe['year'] - 1
        return dataframe

    def covid_adjustment(self, dataframe : pd.DataFrame, target_col = 'MEMS7_ALL'):
        PRE_COVID = dataframe[dataframe['year'] < 2020]
        seasonal_average = PRE_COVID.groupby('month')[target_col].mean().reset_index()
        seasonal_average = seasonal_average.rename(columns = {target_col: 'seasonal_average'})
        dataframe = dataframe.merge(seasonal_average, on = 'month', how = 'left')
        IS_COVID = (((dataframe['year'] == 2020) & (dataframe['month'].isin([3, 4, 5, 11, 12]))) |
                    ((dataframe['year'] == 2021) & (dataframe['month'].isin([1, 2, 3]))))
        dataframe.loc[IS_COVID, target_col] = dataframe.loc[IS_COVID, 'seasonal_average']
        return dataframe.drop(columns = ['seasonal_average'])