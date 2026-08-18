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

            final.append([borough, df_borough_train, df_borough_test, forecast, self.calculate_mae(df_borough_test, forecast)])
        self.sarima_forecast = pd.DataFrame(final, columns=[self.group_col, 'y_train', 'y_test', 'y_forecast', 'mae'])
        return self.sarima_forecast


    def calculate_mae(self, y_test, y_forecast):
        return np.mean(abs(np.array(y_test) - y_forecast[:len(np.array(y_test))]) / np.array(y_test)) * 100

    def get_mae(self, toget='prophet'):
        if toget == 'prophet':
            return np.mean(self.prophet_forecast['mae'])
        elif toget == 'sarima':
            return np.mean(self.sarima_forecast['mae'])
        elif toget == 'bayesian_ridge':
            return np.mean(self.bayesian_ridge_forecast['mae'])
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
            final.append([borough, df_borough_train.values, df_borough_test.values, y_forecast, y_lower, y_upper, self.calculate_mae(df_borough_test.values, y_forecast)])

        self.prophet_forecast = pd.DataFrame(final, columns=[self.group_col, 'y_train', 'y_test', 'y_forecast', 'y_lower', 'y_upper', 'mae'])
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

            final.append([borough, y_train, y_test, y_forecast, e_forecast, self.calculate_mae(y_test, y_forecast)])

        self.bayesian_ridge_forecast = pd.DataFrame(final, columns=[self.group_col, 'y_train', 'y_test', 'y_forecast', 'e_forecast', 'mae'])
        return self.bayesian_ridge_forecast


    def plot_four(self, toplot='prophet', UNCERTAINTY=True):
        if toplot == 'prophet':
            data = self.prophet_forecast
        elif toplot == 'sarima':
            data = self.sarima_forecast
        elif toplot == 'bayesian_ridge':
            data = self.bayesian_ridge_forecast

        n_cols = 4
        n_rows = math.ceil(len(data) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4), sharey=True)

        all_vals = np.concatenate([np.concatenate([np.array(r['y_train']), np.array(r['y_forecast'])]) for _, r in data.iterrows()])
        global_min, global_max = all_vals.min(), all_vals.max()

        boroughs_to_show = ['Barking and Dagenham', 'Barnet', 'Hackney', 'Wandsworth']
        data = data[data[self.group_col].isin(boroughs_to_show)]

        for ax, (idx, row) in zip(axes.flatten(), data.iterrows()):
            vmin = np.percentile(all_vals, 2)
            vmax = np.percentile(all_vals, 98)
            y_train = np.array(row['y_train'])
            y_forecast = np.array(row['y_forecast'])
            t_train = np.arange(len(y_train))
            t_forecast = np.arange(self.t_train_cutoff, self.t_train_cutoff + len(y_forecast))

            train_colors = plt.cm.YlOrRd_r((y_train - vmin) / (vmax - vmin))
            forecast_colors = plt.cm.YlOrRd_r((y_forecast - vmin) / (vmax - vmin))

            ax.plot(np.concatenate([t_train, t_forecast]),
                    np.concatenate([y_train, y_forecast]),
                    color='#00BCD4', linewidth=1.5, alpha=0.7, zorder=1)

            ax.scatter(t_train, y_train, c=train_colors, s=55, zorder=5, edgecolors='none')
            ax.scatter(t_forecast, y_forecast, c=forecast_colors, s=55, zorder=5, edgecolors='none')
            ax.plot(t_train, y_train, color='black', linewidth=1, alpha=0.6, zorder=2)

            ax.set_title(row[self.group_col], fontweight='bold', fontsize=10)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average MEMS')
            ax.set_xticks(self.ticks)
            ax.set_xticklabels(self.labels, rotation=45, fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax in axes.flatten()[len(data):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot(self, toplot='prophet', UNCERTAINTY=True):
            if toplot == 'prophet':
                data = self.prophet_forecast
            elif toplot == 'sarima':
                data = self.sarima_forecast
            elif toplot == 'bayesian_ridge':
                data = self.bayesian_ridge_forecast
    
            n_cols = 4
            n_rows = math.ceil(len(data) / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    
            all_vals = np.concatenate([np.concatenate([np.array(r['y_train']), np.array(r['y_forecast'])]) for _, r in data.iterrows()])
            global_min, global_max = all_vals.min(), all_vals.max()
    
            data = data[data[self.group_col].isin(boroughs_to_show)]
    
            for ax, (idx, row) in zip(axes.flatten(), data.iterrows()):
                vmin = np.percentile(all_vals, 2)
                vmax = np.percentile(all_vals, 98)
                y_train = np.array(row['y_train'])
                y_forecast = np.array(row['y_forecast'])
                t_train = np.arange(len(y_train))
                t_forecast = np.arange(self.t_train_cutoff, self.t_train_cutoff + len(y_forecast))
    
                train_colors = plt.cm.YlOrR_r((y_train - vmin) / (vmax - vmin))
                forecast_colors = plt.cm.YlOrRd_r((y_forecast - vmin) / (vmax - vmin))
    
                ax.plot(np.concatenate([t_train, t_forecast]),
                        np.concatenate([y_train, y_forecast]),
                        color='#00BCD4', linewidth=1.5, alpha=0.7, zorder=1)
    
                ax.scatter(t_train, y_train, c=train_colors, s=55, zorder=5, edgecolors='none')
                ax.scatter(t_forecast, y_forecast, c=forecast_colors, s=55, zorder=5, edgecolors='none')
    
                ax.set_title(row[self.group_col], fontweight='bold', fontsize=10)
                ax.set_xlabel('Year')
                ax.set_ylabel('Average MEMS')
                ax.set_xticks(self.ticks)
                ax.set_xticklabels(self.labels, rotation=45, fontsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
    
            for ax in axes.flatten()[len(data):]:
                ax.set_visible(False)
    
            plt.tight_layout()
            plt.show()