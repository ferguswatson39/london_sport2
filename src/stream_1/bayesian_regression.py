from prepare_boroughs import prepare_borough_data
from sklearn.linear_model import BayesianRidge
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bayesian_ridge_forecast_borough(included_years : list, forecast_years: list, target_col: str) -> pd.DataFrame:
    borough_df = prepare_borough_data()
    # Handle year conversion from e.g: 2016/17 -> 2016 and select years
    borough_df['year'] = borough_df['year'].str.split('/').str[0].astype(int)
    borough_df = borough_df[borough_df['year'].isin(included_years)]
    output = []
    for borough in borough_df['LA_Name'].unique():
        X_train = (np.array(included_years) - 2016).reshape(-1, 1)
        X_forecast = (np.array(forecast_years) - 2016).reshape(-1, 1)
        Y_train = borough_df[borough_df['LA_Name'] == borough][target_col].values
        model = BayesianRidge()
        model.fit(X_train, Y_train)
        Y_forecast, E_forecast = model.predict(X_forecast, return_std = True)    
        years_full = included_years + forecast_years
        participation_full = np.concat((Y_train, Y_forecast))
        error_full = np.concat((np.zeros(len(Y_train)), E_forecast))
        for year, participation, error in zip(years_full, participation_full, error_full):
            output.append({'borough': borough, 'year' : year, 'participation' : participation, 'error' : error})
    return pd.DataFrame(output)

