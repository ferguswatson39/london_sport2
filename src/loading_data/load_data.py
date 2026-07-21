from pathlib import Path
import pandas as pd
import sys
import numpy as np
import pyreadstat
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

def get_2022_data() -> pd.DataFrame:
    dc = DataCatalogue()
    vars = dc.get_perturbation_df_vars()
    vars = vars + ['LondInOut', 'MEMS7_ALL']
    final_vars = [var for var in vars if var != 'active']
    print(final_vars)
    # vars = ['Gend3','HHLiv12','motivd_POP','Motiva_POP','WorkStat10','NSSEC5','Eth7','Age9', 'happy', 'lifesat', 'worthw', 'lone', 'Educ6', 'IMD10','MEMS7_ALL','LondInOut', 'comm1', 'inclus_a']
    df_path = Path(r"C:/Masters/London Sport/9288_ActiveLifeSurvey_2022_2023/UKDA-9288-spss/spss/spss28/active_lives_survey_nov_22-23_data_year_8_shared_20250103.sav")
    df, meta = pyreadstat.read_sav(df_path, usecols = final_vars)
    df = df[df['LondInOut'].notna()]
    # df['LOG_MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])
    missing = [var for var in final_vars if var not in df.columns]
    if len(missing):
        raise KeyError(f'Missing Variables:\n {missing}')
    df['active'] = df['MEMS7_ALL'] >= 150
    # Removed LondInOut and MEMS7_ALL col after it has been used to filter df and create active
    df = df.drop(columns=(['LondInOut', 'MEMS7_ALL']))
    df = df.dropna()
    print(f'DataFrame Cleaned Successfully...')
    print('DataFrame Information:')
    print(f'>>> Columns:\n{df.columns}')
    print(f'>>> Shape {df.shape}')
    return df


