from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from pmdarima import auto_arima
from tqdm.auto import tqdm

def calculate_mape(y_test, y_forecast):
    return np.mean(abs(np.array(y_test) - y_forecast[:len(np.array(y_test))]) / np.array(y_test)) * 100

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

def tune_sarima(df, t_train_cutoff, forecast_steps):
    final = []
    df = df.copy()
    df['t'] = df.groupby('LA_Name').cumcount()
    if 'month' in df.columns:
        seasonal_period = 12
    elif 'quarter' in df.columns:
        seasonal_period = 4
    else:
        seasonal_period = 1

    print("-------------------------")
    print("----- TUNING SARIMA -----")
    print("-------------------------")
    for borough in tqdm(df['LA_Name'].unique()):
        df_borough = df[df['LA_Name'] == borough]

        df_borough_train = df_borough[df_borough['t'] < t_train_cutoff]['MEMS7_ALL']
        df_borough_test = df_borough[df_borough['t'] >= t_train_cutoff]['MEMS7_ALL']
        
        arima_model = auto_arima(df_borough_train, seasonal=True, m=seasonal_period, stepwise=True, suppress_warnings=True, error_action='ignore')
        print(f"{borough}: {arima_model.order} x {arima_model.seasonal_order}")
        forecast = arima_model.predict(n_periods=forecast_steps)
        # borough, train, test, forecast results
        final.append([borough, df_borough_train.values, df_borough_test.values, forecast, calculate_mape(df_borough_test.values, forecast)])
    
    return pd.DataFrame(final, columns=['borough', 'y_train', 'y_test', 'y_forecast', 'mape'])



def bayesian_ridge_forecast(df, t_train_cutoff, forecast_steps):
    final = []

    df = df.copy()
    df['t'] = df.groupby('LA_Name').cumcount()
    for borough in df['LA_Name'].unique():
        df_borough = df[df['LA_Name'] == borough]
        X_train = df_borough[df_borough['t'] < t_train_cutoff]['t'].values.reshape(-1, 1)
        X_forecast = np.arange(t_train_cutoff, t_train_cutoff + forecast_steps).reshape(-1, 1)
        #X_test = df_borough[df_borough['t'] >= t_train_cutoff]['t'].values.reshape(-1, 1)
        y_train = df_borough[df_borough['t'] < t_train_cutoff]['MEMS7_ALL'].values
        y_test = df_borough[df_borough['t'] >= t_train_cutoff]['MEMS7_ALL'].values

        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_forecast, e_forecast = model.predict(X_forecast, return_std = True)    

        final.append([borough, y_train, y_test, y_forecast, e_forecast, calculate_mape(y_test, y_forecast)])
    return pd.DataFrame(final, columns=['borough', 'y_train', 'y_test', 'y_forecast', 'e_forecast', 'mape'])


def plot_borough_forecasts(df, t_train_cutoff, xtick_positions, xtick_labels, UNCERTAINTY=False):
    n_boroughs = len(df)
    n_cols = 4
    n_rows = n_boroughs // 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24), sharey=True)
    for ax, (idx, row) in zip(axes.flatten(), df.iterrows()):
        t_train = range(len(row['y_train']))
        t_test = range(t_train_cutoff, t_train_cutoff + len(row['y_test']))
        t_forecast = range(t_train_cutoff, t_train_cutoff + len(row['y_forecast']))
        t_all = list(t_train) + list(t_test)
        y_all = np.concatenate([row['y_train'], row['y_test']])
        ax.plot(t_all, y_all, color='black')
        ax.plot(t_forecast, row['y_forecast'], color='blue')
        ax.scatter(t_forecast, row['y_forecast'], color='black', s=7)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, fontsize=7)
        ax.text(0.05, 0.95, f"MAPE: {row['mape']:.1f}%",fontsize=7, transform=ax.transAxes, verticalalignment='top')
        ax.set_title(row['borough'], fontweight='bold', fontsize=8)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


