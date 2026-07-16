from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(df, t_train_cutoff, forecast_steps):
    if 'month' in df.columns:
        seasonal_period = 12
    elif 'quarter' in df.columns:
        seasonal_period = 4
    else:
        seasonal_period = 1
    
    model = SARIMAX(df['MEMS7_ALL'], order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))

    results = model.fit()
    forecast_values = results.forecast(forecast_steps).values
    

