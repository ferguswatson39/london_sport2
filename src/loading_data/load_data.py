from pathlib import Path
import pandas as pd
import sys
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.data_catalogue import DataCatalogue

def get_data() -> pd.DataFrame:
    dc = DataCatalogue()
    vars = dc.get_vars()
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    df = pd.read_csv(df_path)
    df['VolAny'] = df['VolAny'].fillna(0.0) # Assumes that any respondent who did not fill voluneering question did not volunteer
    df['active'] = df['MEMS7_ALL'] >= 150
    df = df[df['NSSEC5'] != 5.0]
    df['LOG_MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])
    missing = [var for var in vars if var not in df.columns]
    if len(missing) > 0:
        raise KeyError(f'Missing Variables:\n {missing}')
    df = df[vars].dropna()
    print(f'DataFrame Cleaned Successfully...')
    print('DataFrame Information:')
    print(f'>>> Columns:\n{df.columns}')
    print(f'>>> Shape {df.shape}')
    return df

def get_geographic_data() -> pd.DataFrame:
    dc = DataCatalogue()
    geographic = dc.get_geographic_vars()
    targets = dc.get_target_vars()
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    df = pd.read_csv(df_path)
    df['LOG_MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])
    df['active'] = df['MEMS7_ALL'] >= 150
    df = df[geographic + targets + ['year', 'month']]
    df['LA_Name'] = df['LA_2023'].map(dc.get_data_dict()['LA_2023']['value_labels'])       
    print(f'DataFrame Cleaned Successfully...')
    print('DataFrame Information:')
    print(f'>>> Columns:\n{df.columns}')
    print(f'>>> Shape {df.shape}')
    return df

