from pathlib import Path
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.data_catalogue import DataCatalogue

def get_data() -> pd.DataFrame:
    dc = DataCatalogue()
    vars = dc.get_vars()
    df_path = Path(ROOT / 'exploration' / 'data' / 'master_data' / '2015_to_2023_full_preprocessed_data_set.csv.gz')
    df = pd.read_csv(df_path)
    df['VolAny'] = df['VolAny'].fillna(0.0) # Assumes that any respondent who did not fill voluneering question did not volunteer
    df['active'] = df['MEMS7_ALL'] >= 150
    df = df[df['NSSEC5'] != 5.0]
    missing = [var for var in vars if var not in df.columns]
    if len(missing) > 0:
        raise KeyError(f'Missing Variables:\n {missing}')
    df = df[vars].dropna()
    print(f'DataFrame Cleaned Successfully...')
    print('DataFrame Information:')
    print(f'>>> Columns:\n{df.columns}')
    print(f'>>> Shape {df.shape}')
    return df