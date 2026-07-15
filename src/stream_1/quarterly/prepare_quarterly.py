from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_geographic_data
from src.loading_data.data_catalogue import DataCatalogue

def prepare_quarterly_boroughs():
    dc = DataCatalogue()
    df_path = Path(ROOT / 'exploration' / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    raw = pd.read_csv(df_path)

    print(raw[raw['year'] == '2022/23']['month'].value_counts())

    raw['month'] = ((raw['month'] - 3) % 12) + 1
    raw['year'] = raw['year'].str.split('/').str[1].astype(int) + 2000
    raw.loc[raw['month'].isin([11,12]), 'year'] = raw['year'] - 1  

    #raw['quarter'] = pd.cut(raw['month'], bins=[0,3,6,9,12], labels=[1,2,3,4])
    raw['LA_Name'] = raw['LA_2023'].map(dc.get_data_dict()['LA_2023']['value_labels'])

    borough_df = raw.groupby(['LA_2023', 'LA_Name', 'year', 'month']).agg(
        MEMS7_ALL=('MEMS7_ALL', 'mean'),
        IMD_mean=('IMD10', 'mean'),
        LondInOut=('LondInOut', 'first')
    ).reset_index().sort_values(['LA_2023', 'year', 'month']).reset_index(drop=True)
    borough_df['LA_Name'] = borough_df['LA_Name'].replace({
        'Kingston upon Thames': 'Kingston',
        'Richmond upon Thames': 'Richmond'
    })
    borough_df['LondInOut'] = borough_df['LondInOut'].map({1.0: 'Outer', 2.0: 'Inner'})
    borough_df = borough_df[borough_df['LA_2023'] != 59.0]
    borough_df.sort_values(['LA_2023', 'year', 'month'], inplace=True)
    return borough_df