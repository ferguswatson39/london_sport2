from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np

def calculate_mape(y_test, y_forecast):
    return np.mean(abs(y_test - y_forecast[:len(y_test)]) / y_test) * 100

def sarima_forecast(df, t_train_cutoff, forecast_steps):
    final = []
    df = df.copy()
    df['t'] = df.groupby('LA_Name').cumcount()
    if 'month' in df.columns:
        seasonal_period = 12
    elif 'quarter' in df.columns:
        seasonal_period = 4
    else:
        seasonal_period = 1

    for borough in df['LA_Name'].unique():
        df_borough = df[df['LA_Name'] == borough]

        df_borough_train = df_borough[df_borough['t'] < t_train_cutoff]['MEMS7_ALL']
        df_borough_test = df_borough[df_borough['t'] >= t_train_cutoff]['MEMS7_ALL']
        
        model = SARIMAX(df_borough_train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))

        results = model.fit(disp=False)
        forecast = results.forecast(forecast_steps).values
        # borough, train, test, forecast results
        final.append([borough, df_borough_train, df_borough_test, forecast, calculate_mape(df_borough_test, forecast)])
    
    return pd.DataFrame(final, columns=['borough', 'y_train', 'y_test', 'y_forecast', 'mape'])
