from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_data
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def check_num_obs(X, year):
    no_obs = []
    num_zeroes = (X[X== 0].count()/len(X)).sort_values(ascending=False) 
    for idx, value in enumerate(num_zeroes):
        if value == 1.0:
            no_obs.append(num_zeroes.index[idx])
    if len(no_obs) > 0:
        print(f'Year: {year}. Columns found to have zero observations:')
        print(no_obs)
        return no_obs
    else:
        print(f'Year: {year}. All columns have observations')
        return []
    
def generate_coeficients() -> pd.DataFrame:
    df = get_data()
    dont_encode = ['nchild', 'active', 'MEMS7_ALL', 'year']
    discrete_variables = [c for c in df.columns if c not in dont_encode]
    reference_categories = [f'{c}_{df[c].value_counts().index[0]}' for c in discrete_variables]
    encoder = OneHotEncoder(drop='first', handle_unknown = 'ignore', sparse_output = False).set_output(transform='pandas')
    encoded = encoder.fit_transform(df[discrete_variables])
    df_encoded = pd.concat([df, encoded], axis=1).drop(columns=discrete_variables+['MEMS7_ALL']) # dropped gender= other as coef=0
    all_years = []
    for year in sorted(df_encoded['year'].unique()):
        df_encoded_sy = df_encoded[df_encoded['year'] == year]
        df_encoded_sy = df_encoded_sy.drop(columns=['year'])
        Y = df_encoded_sy['active']
        X = df_encoded_sy.drop(columns=['active'])
        scaler = StandardScaler()
        continuous_cols = ['nchild']
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
        no_obs = check_num_obs(X, year)
        if len(no_obs) > 0:
            X = X.drop(columns = no_obs)
        logistic = sm.Logit(Y, X)
        output = logistic.fit()
        logistic_results = pd.DataFrame({
            'feature_names' : output.params.index,
            'odds_ratios' : np.exp(output.params.values),
            'pvalues' : output.pvalues.values,
            'confidence_lower' : np.exp(output.conf_int()[0].values),
            'confidence_upper' : np.exp(output.conf_int()[1].values),
            'std_error' : output.bse.values,
            'year' : year 
        })
        all_years.append(logistic_results)
    all = pd.concat(all_years)
    save_path = Path(ROOT / 'data' / 'logistic_coefficients')
    all.to_csv(save_path / 'logistic_coef_results.csv', index=False)
    print(f'Saved Logistic Coefficient Results to\n:>>>{save_path}')
    return 'done'

# generate_coeficients()