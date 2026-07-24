from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from prophet import Prophet
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from pmdarima import auto_arima
from tqdm.auto import tqdm


class Forecast:
    def __init__(self, df, forecast_steps, group_col='LA_Name'):
        self.df = df
        self.sarima_forecast = None
        self.prophet_forecast = None
        self.bayesian_ridge_forecast = None
        self.group_col = group_col
        self.labels = ['2017', '2018', '2019', '2020', '2021', '2022']
        self.forecast_steps = forecast_steps
        self.seasonal_period = self._set_seasonal_period()

    def _set_seasonal_period(self):
        if 'month' in self.df.columns:
            self.t_train_cutoff = 72
            self.ticks = [2, 14, 26, 38, 50, 62]
            
            for i in range(self.forecast_steps // 12):
                self.labels.append(str(2023 + i))
                self.ticks.append(i*12 + 72)
            return 12
        elif 'quarter' in self.df.columns:
            self.t_train_cutoff = 22
            self.ticks = [0, 4, 8, 12, 16, 20]
            for i in range(self.forecast_steps // 4):
                self.labels.append(str(2023 + i))
                self.ticks.append(i*4 + 24)
            return 4
        else:
            self.t_train_cutoff = 6
            self.ticks = [1,2,3,4,5,6]
            for i in range(self.forecast_steps):
                self.labels.append(str(2023 + i))
                self.ticks.append(i + 7)
            return 1

    def sarima(self):
        if self.seasonal_period == 1:
            return None
        final = []
        df = self.df.copy()
        df['t'] = df.groupby(self.group_col).cumcount()

        for borough in df[self.group_col].unique():
            df_borough = df[df[self.group_col] == borough]

            df_borough_train = df_borough[df_borough['t'] < self.t_train_cutoff]['MEMS7_ALL']
            df_borough_test = df_borough[df_borough['t'] >= self.t_train_cutoff]['MEMS7_ALL']
            
            model = SARIMAX(df_borough_train, order=(1,1,1), seasonal_order=(1,1,1,self.seasonal_period))
            results = model.fit(disp=False)
            forecast = results.forecast(self.forecast_steps).values

            final.append([borough, df_borough_train, df_borough_test, forecast, self.calculate_mape(df_borough_test, forecast)])
        self.sarima_forecast = pd.DataFrame(final, columns=[self.group_col, 'y_train', 'y_test', 'y_forecast', 'mape'])
        return self.sarima_forecast


    def calculate_mape(self, y_test, y_forecast):
        return np.mean(abs(np.array(y_test) - y_forecast[:len(np.array(y_test))]) / np.array(y_test)) * 100

    def get_mape(self, toget='prophet'):
        if toget == 'prophet':
            return np.mean(self.prophet_forecast['mape'])
        elif toget == 'sarima':
            return np.mean(self.sarima_forecast['mape'])
        elif toget == 'bayesian_ridge':
            return np.mean(self.bayesian_ridge_forecast['mape'])
        else:
            return -1


    def prophet(self):
        final = []
        df = self.df.copy()
        df['t'] = df.groupby(self.group_col).cumcount()

        for borough in df[self.group_col].unique():
            df_borough = df[df[self.group_col] == borough]

            if 'month' in df.columns:
                dates = pd.to_datetime(df_borough[['year', 'month']].assign(day=1))
                freq = 'MS'
            elif 'quarter' in df.columns:
                month_num = df_borough['quarter'].astype(int) * 3 - 2
                dates = pd.to_datetime(df_borough[['year']].assign(month=month_num, day=1))
                freq = 'QS'
            else:
                dates = pd.to_datetime(df_borough[['year']].assign(month=1, day=1))
                freq = 'YS'

            df_prophet = pd.DataFrame({'ds': dates, 'y': df_borough['MEMS7_ALL'].values})
            df_prophet_train = df_prophet[df_prophet['ds'] < dates.iloc[self.t_train_cutoff]]

            df_borough_train = df_borough[df_borough['t'] < self.t_train_cutoff]['MEMS7_ALL']
            df_borough_test = df_borough[df_borough['t'] >= self.t_train_cutoff]['MEMS7_ALL']

            covid = pd.DataFrame({
                'holiday': 'covid',
                'ds': pd.to_datetime(['2020-01-01', '2021-01-01']),
                'lower_window': 0,
                'upper_window': 1
            })

            model = Prophet(holidays=covid)
            
            model = Prophet(yearly_seasonality=True, holidays=covid, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet_train)
            future = model.make_future_dataframe(periods=self.forecast_steps, freq=freq)
            forecast_df = model.predict(future)
            y_forecast = forecast_df['yhat'].values[-self.forecast_steps:]
            y_lower = forecast_df['yhat_lower'].values[-self.forecast_steps:]
            y_upper = forecast_df['yhat_upper'].values[-self.forecast_steps:]
            # borough, train, test, forecast results
            final.append([borough, df_borough_train.values, df_borough_test.values, y_forecast, y_lower, y_upper, self.calculate_mape(df_borough_test.values, y_forecast)])

        self.prophet_forecast = pd.DataFrame(final, columns=[self.group_col, 'y_train', 'y_test', 'y_forecast', 'y_lower', 'y_upper', 'mape'])
        return self.prophet_forecast


    def bayesian_ridge(self):
        final = []

        df = self.df.copy()
        df['t'] = df.groupby(self.group_col).cumcount()
        for borough in df[self.group_col].unique():
            df_borough = df[df[self.group_col] == borough]
            X_train = df_borough[df_borough['t'] < self.t_train_cutoff]['t'].values.reshape(-1, 1)
            X_forecast = np.arange(self.t_train_cutoff, self.t_train_cutoff + self.forecast_steps).reshape(-1, 1)
            #X_test = df_borough[df_borough['t'] >= t_train_cutoff]['t'].values.reshape(-1, 1)
            y_train = df_borough[df_borough['t'] < self.t_train_cutoff]['MEMS7_ALL'].values
            y_test = df_borough[df_borough['t'] >= self.t_train_cutoff]['MEMS7_ALL'].values

            model = BayesianRidge()
            model.fit(X_train, y_train)
            y_forecast, e_forecast = model.predict(X_forecast, return_std = True)    

            final.append([borough, y_train, y_test, y_forecast, e_forecast, self.calculate_mape(y_test, y_forecast)])

        self.bayesian_ridge_forecast = pd.DataFrame(final, columns=[self.group_col, 'y_train', 'y_test', 'y_forecast', 'e_forecast', 'mape'])
        return self.bayesian_ridge_forecast


    def plot(self, toplot='prophet', UNCERTAINTY=False):
        if toplot == 'prophet':
            data = self.prophet_forecast
            UNCERTAINTY = True
        elif toplot == 'sarima':
            data = self.sarima_forecast
        elif toplot == 'bayesian_ridge':
            data = self.bayesian_ridge_forecast

        if self.group_col == 'LA_Name':
            n_boroughs = len(data)
            n_cols = 4
            n_rows = n_boroughs // 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
            for ax, (idx, row) in zip(axes.flatten(), data.iterrows()):
                t_train = range(len(row['y_train']))
                t_test = range(self.t_train_cutoff, self.t_train_cutoff + len(row['y_test']))
                t_forecast = range(self.t_train_cutoff, self.t_train_cutoff + len(row['y_forecast']))
                t_all = list(t_train) + list(t_test)
                y_all = np.concatenate([row['y_train'], row['y_test']])
                ax.plot(t_all, y_all, color='black')
                ax.plot(t_forecast, row['y_forecast'], color='blue')
                ax.scatter(t_forecast, row['y_forecast'], color='black', s=7)
                if UNCERTAINTY: ax.fill_between(t_forecast, row['y_lower'], row['y_upper'], alpha=0.2, color='blue')
                ax.set_xticks(self.ticks)
                ax.set_ylim(0, 1500)
                ax.set_xticklabels(self.labels, rotation=45, fontsize=7)
                ax.text(0.05, 0.95, f"MAPE: {row['mape']:.1f}%",fontsize=7, transform=ax.transAxes, verticalalignment='top')
                ax.set_title(row[self.group_col], fontweight='bold', fontsize=8)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.show()

        else:
            n_boroughs = len(data)
            n_cols = 3
            n_rows = math.ceil(n_boroughs / 3)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
            for ax, (idx, row) in zip(axes.flatten(), data.iterrows()):
                t_train = range(len(row['y_train']))
                t_test = range(self.t_train_cutoff, self.t_train_cutoff + len(row['y_test']))
                t_forecast = range(self.t_train_cutoff,self.t_train_cutoff + len(row['y_forecast']))
                t_all = list(t_train) + list(t_test)
                y_all = np.concatenate([row['y_train'], row['y_test']])
                ax.plot(t_all, y_all, color='black')
                ax.plot(t_forecast, row['y_forecast'], color='blue')
                ax.scatter(t_forecast, row['y_forecast'], color='black', s=7)
                if UNCERTAINTY: ax.fill_between(t_forecast, row['y_lower'], row['y_upper'], alpha=0.2, color='blue')
                ax.set_xticks(self.ticks)
                ax.set_ylim(0, max(y_all) * 1.3)
                ax.set_xticklabels(self.labels, rotation=45, fontsize=7)
                ax.text(0.05, 0.95, f"MAPE: {row['mape']:.1f}%",fontsize=7, transform=ax.transAxes, verticalalignment='top')
                ax.set_title(f'Cluster {row[self.group_col]}', fontweight='bold', fontsize=8)
                ax.spines['right'].set_visible(False)
            for ax in axes.flatten()[len(data):]:
                ax.set_visible(False)
            plt.tight_layout()
            plt.show()