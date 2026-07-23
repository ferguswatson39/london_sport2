from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_geographic_data
from src.loading_data.data_catalogue import DataCatalogue
from covid_analysis import Covid

def prepare_borough_data(target_col = 'MEMS7_ALL', ADJUST_COVID = True):
    covid = Covid()
    borough_df = get_geographic_data()
    borough_df = covid.correct_year_and_month(borough_df)
    borough_df[target_col] = borough_df[target_col].astype(float)
    if ADJUST_COVID:
        adjusted = []
        for borough in borough_df['LA_Name'].unique():
            borough_adjusted = covid.covid_adjustment(borough_df[borough_df['LA_Name'] == borough], target_col = target_col)
            adjusted.append(borough_adjusted)
        borough_df = pd.concat(adjusted, ignore_index = True)
    borough_df = borough_df.groupby(['LA_2023', 'LA_Name', 'year'])[target_col].mean().reset_index()
    borough_df['LA_Name'] = borough_df['LA_Name'].replace({'Kingston upon Thames': 'Kingston', 'Richmond upon Thames': 'Richmond'})
    borough_df = borough_df[borough_df['LA_2023'] != 59.0] # Removing City of London (COVID Outliers)
    return borough_df

def prepare_national_data(target_col = 'MEMS7_ALL', ADJUST_COVID = True):
    covid = Covid()
    national_df = get_geographic_data()
    national_df = covid.correct_year_and_month(national_df)
    if ADJUST_COVID: national_df = covid.covid_adjustment(national_df, target_col = target_col)
    national_df = national_df.groupby('year')[target_col].mean()
    national_df = national_df.reset_index()
    return national_df

def prepare_grouped_boroughs():
    dc = DataCatalogue()
    df_path = Path(ROOT / 'exploration' / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    raw = pd.read_csv(df_path)
    raw['LA_Name'] = raw['LA_2023'].map(dc.get_data_dict()['LA_2023']['value_labels'])
    borough_df = raw.groupby(['LA_2023', 'LA_Name', 'year']).agg(
        MEMS7_ALL=('MEMS7_ALL', 'mean'),
        IMD_mean=('IMD10', 'mean'),
        LondInOut=('LondInOut', 'first')
    ).reset_index()
    borough_df['LA_Name'] = borough_df['LA_Name'].replace({
        'Kingston upon Thames': 'Kingston',
        'Richmond upon Thames': 'Richmond'
    })
    borough_df['LondInOut'] = borough_df['LondInOut'].map({1.0: 'Outer', 2.0: 'Inner'})
    borough_df = borough_df[borough_df['LA_2023'] != 59.0]
    print('DataFrame Cleaned Successfully...')
    print(f'>>> Columns:\n{borough_df.columns.tolist()}')
    print(f'>>> Shape {borough_df.shape}')
    return borough_df

    