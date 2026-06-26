from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent

def get_data() -> pd.DataFrame:
    demographic_cols = ['Gend3', 'Disab3', 'Age9', 'Eth7','NSSEC5']
    interested_cols = [
    'Educ6', 'MEMS7_ALL','year','IMD10', # Depravation
    'nchild', 'nadults', # Childcare responsibilties
    #'health', # General Health - variable is not in 2016/17
    'VolAny', # Has Volunteered
    'motivex2a', 'motivex2b', 'motivex2c', 'motivex2d', # Motivations
    'Motiva_POP', 'motivb_POP', 'motivc_POP', 'motivd_POP','motive_POP', 
    'inclus_a', 'inclus_b', 'inclus_c','comm1', 'comm2', # Social Cohesion
    'anxious',  'happy', 'lifesat', 'lone', 'worthw', # Life Emotions
    'indev', 'indevtry', # Different types of motivations
    'limfreti1', 'limfreti2', 'limfreti3', 'limfreti4',
    'limfreti5', 'limfreti6', 'limfreti7', 'limfreti8', # Limitations on free time
    'WorkStat10'# 'WorkStat5', 'WorkStat7' # Employment Status
    ]
    df_path = Path(ROOT / 'exploration' / 'data' / 'master_data' / '2015_to_2023_full_preprocessed_data_set.csv.gz')
    df = pd.read_csv(df_path)
    df['VolAny'] = df['VolAny'].fillna(0.0) # Assumes that any respondent who did not fill voluneering question did not volunteer
    df['active'] = df['MEMS7_ALL'] > 150
    available = [col for col in interested_cols if col in df.columns]
    unavailable = [col for col in interested_cols if col not in df.columns]
    healthy_cols, unhealthy_cols = missingness(80000, available, df) # Only keeps columns with 80000 / 120000 non missing values
    total_cols = demographic_cols + healthy_cols
    df = df[total_cols].dropna()
    print(f'DataFrame Cleaned Sucessfully...')
    print('DataFrame Information:')
    print(f'>>> Columns:\n{df.columns}')
    print(f'>>> Shape {df.shape}')
    print('--------------------------')
    print(f'Columns NOT in Master.csv:\n>>> {unavailable}')
    print(f'Unhealthy Columns:\n>>> {unhealthy_cols}') 
    return df

def missingness(health :int, available_cols:list, df : pd.DataFrame) -> list[str]:
    healthy_cols = []
    unhealthy_cols = []
    for j in available_cols:
        new = df[j].dropna()
        # print(f'Cols: {j}\nHas Size:{len(new)}')
        if len(new) > health:
            healthy_cols.append(j)
        else:
            unhealthy_cols.append(j)
    return healthy_cols, unhealthy_cols