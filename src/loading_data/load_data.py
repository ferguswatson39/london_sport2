from pathlib import Path
import pandas as pd
import sys
import numpy as np
import pyreadstat
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.data_catalogue import DataCatalogue
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.stream_1.covid_analysis import Covid

SPORT_MAP = {
    'WALKALL' : 'Walking', 'CYCALL' : 'Cycling', 'RUNATHMULTI' : 'Running', 'ADVWATERSPORT' : 'Water Sports',
    'TEAMSPORT' : 'Team Sports', 'DANCEALL' : 'Dance/Gymnastics', 'FOOTBALL' : 'Football', 'RACKETSPORT' : 'Racket Sports',
    'COMBATTARGET' : 'Combat Sports', 'WINTER' : 'Winter Sports', 'CRICKET' : 'Cricket', 'BASKETBALL' : 'Basketball',
    'GYMTRAMPCHEER' : 'Gymnastics', 'RUGBYUNION' : 'Rugby', 'NETBALL' : 'Netball', 'HOCKEY' : 'Hockey'
}

CATEGORY_MAP = {
    'Walking' : 'Active Travel',  'Cycling' : 'Active Travel', 'Running' : 'Active Travel',
    'Water Sports' : 'Water Sports', 'Dance/Gymnastics' : 'Dance/Gymnastics', 'Racket Sports' : 'Racket Sports',
    'Combat Sports' : 'Combat Sports', 'Winter Sports' : 'Winter Sports', 'Team Sports' : 'Team Sports'
}

MONTH_MAP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9 : 3, 10: 4, 11: 4, 12: 4}
QUARTER_MAP = {1: 2, 2: 5, 3: 8, 4: 11}

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

def get_modality_data() -> pd.DataFrame:
    dc = DataCatalogue()
    df_path = Path(ROOT / 'exploration' / 'data' / 'master_data' / '2016_to_2023_master_clustering_data_set.csv')
    df = pd.read_csv(df_path)
    df = df[['LCA_Class', 'LA_2023']]
    df = df.dropna()
    df['LA_Name'] = df['LA_2023'].map(dc.get_data_dict()['LA_2023']['value_labels']) 
    df['LA_Name'] = df['LA_Name'].replace({'Richmond upon Thames': 'Richmond', 'Barking and Dagenham' : 'Barking/Dagenham'})
    crosstab = pd.crosstab(df['LCA_Class'], df['LA_Name'], normalize = 'index') * 100
    modes = crosstab.idxmax(axis = 1)
    percentages = crosstab.max(axis = 1)
    return pd.DataFrame({'LA': modes, 'Percentage' : percentages}).reset_index()

def get_sporting_distributions(significance_threshold : float = 0.5):
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    df = pd.read_csv(df_path)
    MEMS_COLS = [col for col in df.columns if col.startswith('MEMS7_')]
    total_mems = (df['MEMS7_ALL'] * df['wt_final']).sum()
    results = []
    for col in MEMS_COLS:
        mems_contribution = (df[col].fillna(0) * df['wt_final']).sum() 
        results.append({'Sport' : col, 'MEMS Contribution %' : mems_contribution / total_mems * 100})
    distribution_df = pd.DataFrame(results).sort_values(by = 'MEMS Contribution %', ascending = False).reset_index(drop = True)
    distribution_df = distribution_df[distribution_df['MEMS Contribution %'] >= significance_threshold]
    exclude = ['MEMS7_ALL', 'A0', 'B0', 'WALKALLRUN', 'ACTTRAV', 'LEISURE', 'ROLLER', 'F']
    distribution_df = distribution_df[~ distribution_df['Sport'].str.contains('|'.join(exclude))]
    distribution_df['Sport'] = distribution_df['Sport'].str.split('_').str[1].replace(SPORT_MAP)
    distribution_df['Category'] = distribution_df['Sport'].replace(CATEGORY_MAP)
    distribution_df.loc[distribution_df['Sport'] == 'Dance/Gymnastics', 'MEMS Contribution %'] += distribution_df.loc[distribution_df['Sport'] == 'Gymnastics', 'MEMS Contribution %'].sum()
    distribution_df = distribution_df[distribution_df['Sport'] != 'Gymnastics']
    return distribution_df

def get_sports_matrix(minutes_threshold : int = 0):
    df_path = Path(ROOT / 'data' / 'raw_datasets' / 'active_lives_survey_nov_2022_23_data_shared_20250103.sav')
    _, meta = pyreadstat.read_sav(df_path, metadataonly = True)
    exclude = ['MEMS7_ALL', 'A0', 'B0', 'WALKALLRUN', 'ACTTRAV', 'C0', 'C1', 'CAPPED']
    MEMS_COLS = [col for col in meta.column_names if col.startswith('MEMS7_') and not any(ex in col for ex in exclude)] + ['MEMS7_CYCALL_C02', 'MEMS7_WALKALL_C01']
    ALL_COLS = MEMS_COLS + ['LondInOut', 'serial']
    sports_df, _ = pyreadstat.read_sav(df_path, usecols = ALL_COLS)
    sports_df = sports_df.dropna(subset = ['LondInOut'])
    sports_df = sports_df.drop(columns = 'LondInOut')
    sports_df[MEMS_COLS] = (sports_df[MEMS_COLS].fillna(0) > minutes_threshold).astype(int)
    sports_df = sports_df.rename(columns = {col : col.split('_')[1] for col in MEMS_COLS})
    return sports_df

def get_monthly_data(ADJUST_COVID = False):
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    monthly_df = pd.read_csv(df_path)
    covid = Covid()
    monthly_df = covid.correct_year_and_month(monthly_df)
    if ADJUST_COVID: monthly_df = covid.covid_adjustment(monthly_df)
    monthly_df = monthly_df.groupby(['year', 'month'])['MEMS7_ALL'].agg(['mean', 'count']).reset_index()
    monthly_df['date'] = pd.to_datetime(monthly_df['year'].astype(str) + monthly_df['month'].astype(int).astype(str), format='%Y%m')
    return monthly_df

def get_quarterly_data(ADJUST_COVID = False):
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    quarterly_df = pd.read_csv(df_path)
    covid = Covid()
    quarterly_df = covid.correct_year_and_month(quarterly_df)
    quarterly_df['quarter_num'] = quarterly_df['month'].map(MONTH_MAP)
    quarterly_df['quarter_date'] = pd.to_datetime(quarterly_df['year'].astype(str) + quarterly_df['quarter_num'].map(QUARTER_MAP).astype(str), format='%Y%m')
    quarterly_df = quarterly_df[~((quarterly_df['year'] == 2023) & (quarterly_df['quarter_num'] == 4))]
    if ADJUST_COVID: quarterly_df = covid.covid_adjustment(quarterly_df)
    quarterly_df = quarterly_df.groupby(['year', 'quarter_date', 'quarter_num'])['MEMS7_ALL'].agg(['mean', 'count']).reset_index()
    return quarterly_df
    
def get_2022_data() -> pd.DataFrame:
    dc = DataCatalogue()
    vars = dc.get_perturbation_df_vars()
    vars = vars + ['LondInOut', 'MEMS7_ALL']
    final_vars = [var for var in vars if var != 'active']
    print(final_vars)
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

def get_master_2022_data():
    dc = DataCatalogue()
    core_vars = dc.get_perturbation_core()
    all = core_vars + ['serial', 'year', 'LCA_Class', 'MEMS7_ALL']
    print(all)
    path = ROOT / 'data' / 'master_data' / '2016_to_2023_master_clustering_data_set.csv'
    df = pd.read_csv(path, usecols = all)
    missing = [var for var in all if var not in df.columns]
    if missing:
        raise ValueError(f'{missing} MISSING')
    df = df[df['year'] == '2022/23']
    df = df[df['LCA_Class'].notna()]
    df['active'] = df['MEMS7_ALL'] >= 150
    df['NSSEC5'] = df['NSSEC5'].fillna(5)
    missing = df.isna().sum()
    for idx, m in enumerate(missing):
        if m > 0:
            raise ValueError(f'{missing.index[idx]} has missing Values')
    print('Data loaded with zero missing values.')
    return df

def get_raw_2022_data():
    dc = DataCatalogue()
    to_perturb = dc.get_perturbation_vars()
    full_cols = ['serial'] + to_perturb
    data_path = r"C:\Masters\London Sport\9288_ActiveLifeSurvey_2022_2023\UKDA-9288-spss\spss\spss28\active_lives_survey_nov_22-23_data_year_8_shared_20250103.sav"
    df, meta = pyreadstat.read_sav(data_path, usecols = full_cols)
    missing = [var for var in full_cols if var not in df.columns]
    if missing:
        raise ValueError(f'{missing} MISSING')
    return df

def merge_frames(on : str, primary_frame : pd.DataFrame, secondary_frame : pd.DataFrame ) -> pd.DataFrame:
    combined = primary_frame.merge(secondary_frame, how = 'left', on = on )
    return combined

def get_clean_2022():
    primary_frame = get_master_2022_data()
    print('Loaded primary frame...')
    secondary_frame = get_raw_2022_data()
    print('Loaded secondary frame...')
    combined = merge_frames(on = 'serial', primary_frame = primary_frame, secondary_frame = secondary_frame)
    print('Merged Frames.')
    combined = combined.dropna()
    print('Dropped NAN values')
    return combined

def one_hot_encode_frame(df : pd.DataFrame):
    dc = DataCatalogue()
    discrete_vars = dc.get_perturbation_catogs()
    encoder = OneHotEncoder(drop='first', handle_unknown = 'ignore', sparse_output = False).set_output(transform = 'pandas')
    encoded = encoder.fit_transform(df[discrete_vars])
    df_encoded = pd.concat([df, encoded], axis = 1).drop(columns = discrete_vars)
    return df_encoded

def get_master_data():
    dc = DataCatalogue()
    # get_perturbation_core loads in perturbation vers but also includes 'active'
    # need to drop 'active' before goes into usecols
    # core_vars = dc.get_perturbation_core()
    vars = dc.get_clustering_continuous() + dc.get_clustering_categoricals()
    all = vars + ['serial', 'year', 'MEMS7_ALL', 'LCA_Class']
    ##########################################
    ##### Temporary Fix - Drop motiva_pop ####
    ##########################################
    all = [var for var in all if var != 'Motiva_POP']
    path = ROOT / 'data' / 'master_data' / '2016_to_2023_master_clustering_data_set.csv'
    df = pd.read_csv(path, usecols = all)
    missing = [var for var in all if var not in df.columns]
    if missing:
        raise ValueError(f'{missing} MISSING')
    df = df[df['LCA_Class'].notna()]
    df['active'] = df['MEMS7_ALL'] >= 150
    df['NSSEC5'] = df['NSSEC5'].fillna(5)
    missing = df.isna().sum()
    for idx, m in enumerate(missing):
        if m > 0:
            raise ValueError(f'{missing.index[idx]} has missing Values')
    print('Data loaded with zero missing values.')
    return df