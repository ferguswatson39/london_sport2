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
from src.loading_data.data_catalogue import DataCatalogue
from src.stream_2.check_empty_dummies import check_empty_dummies

def generate_reg_coef() -> pd.DataFrame:
    dc = DataCatalogue()
    df = get_data()
    discrete_variables = dc.get_to_be_encoded_vars()
    # reference_categories = [f'{c}_{df[c].value_counts().index[0]}' for c in discrete_variables]
    encoder = OneHotEncoder(drop='first', handle_unknown = 'ignore', sparse_output = False).set_output(transform = 'pandas')
    encoded = encoder.fit_transform(df[discrete_variables])
    df_encoded = pd.concat([df, encoded], axis = 1).drop(columns = discrete_variables)
    all_years = []
    for year in sorted(df_encoded['year'].unique()):
        df_encoded_sy = df_encoded[df_encoded['year'] == year]
        df_encoded_sy = df_encoded_sy.drop(columns=['year'])
        Y = df_encoded_sy['LOG_MEMS7_ALL']
        X = df_encoded_sy.drop(columns=dc.get_target_vars())
        X = sm.add_constant(X) # Need to add constant
        scaler = StandardScaler()
        continuous_cols = dc.get_continuous_vars(with_targets = False)
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
        no_obs = check_empty_dummies(X, year)
        if len(no_obs) > 0:
            X = X.drop(columns = no_obs)
        print(f'Columns into Model:\n>>> Y = {Y.name}\n>>> X = {X.columns}')
        ols = sm.OLS(Y, X)
        output = ols.fit()
        print(output.summary())
        ols_results = pd.DataFrame({
            'feature_names' : output.params.index,
            'coef' : output.params.values,
            'coef_as_percent' : ((np.exp(output.params.values) - 1) * 100),
            'pvalues' : output.pvalues.values,
            'confidence_lower' : output.conf_int()[0].values,
            'confidence_upper' : output.conf_int()[1].values,
            'std_error' : output.bse.values,
            'rsquared' : output.rsquared,
            'adj_rsquared' : output.rsquared_adj,
            'year' : year 
        })
        all_years.append(ols_results)
    all = pd.concat(all_years)
    save_path = Path(ROOT / 'data' / 'ols_coefficients')
    all.to_csv(save_path / 'ols_coef_results.csv', index=False)
    print(f'Saved OLS Coefficient Results to\n:>>>{save_path}')
    return 'done'

generate_reg_coef()