from prepare_boroughs import prepare_borough_data
from typing import List
from sklearn.linear_model import BayesianRidge
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BayesianBoroughForecaster:
    def __init__(self, 
                 INCLUDED_YEARS : List[int] = [2017, 2018, 2019, 2020, 2021, 2022, 2023],
                 FORECAST_YEARS : List[int] = [2024, 2025, 2026, 2027],
                 TEST_YEARS : List[int] = [2022, 2023],
                 target_col: str = 'MEMS7_ALL',
                 ADJUST_COVID: bool = False
    ):
        self.INCLUDED_YEARS = INCLUDED_YEARS
        self.FORECAST_YEARS = FORECAST_YEARS
        self.TEST_YEARS = TEST_YEARS
        self.target_col = target_col
        self.ADJUST_COVID = ADJUST_COVID
        self.forecast_df : pd.DataFrame = None

    def sort_and_group_boroughs(self) -> pd.DataFrame:
        borough_df = prepare_borough_data(ADJUST_COVID = self.ADJUST_COVID, target_col = self.target_col)
        borough_df = borough_df[borough_df['year'].isin(self.INCLUDED_YEARS)]
        borough_df = borough_df.sort_values(by = ['LA_Name', 'year']).reset_index(drop = True)
        borough_df[self.target_col] = borough_df.groupby('LA_Name')[self.target_col].ffill() # Important to avoid future leakage
        return borough_df

    def fit_predict(self) -> pd.DataFrame:
        borough_df = self.sort_and_group_boroughs()
        output = []
        for borough in borough_df['LA_Name'].unique():
            X_train = (np.array(self.INCLUDED_YEARS) - 2017).reshape(-1, 1)
            X_forecast = (np.array(self.FORECAST_YEARS) - 2017).reshape(-1, 1)
            Y_train = borough_df[borough_df['LA_Name'] == borough][self.target_col].values
            X_full = np.concat((X_train, X_forecast))
            model = BayesianRidge()
            model.fit(X_train, Y_train)
            Y_forecast, E_forecast = model.predict(X_forecast, return_std = True)    
            Y_target = np.concat((Y_train, Y_forecast))
            Y_predicted = model.predict(X_full) 
            error_full = np.concat((np.zeros(len(Y_train)), E_forecast * 2)) # 95 % Confidence Interval
            for year, target, predicted, error in zip(self.INCLUDED_YEARS + self.FORECAST_YEARS, Y_target, Y_predicted, error_full):
                output.append({'borough': borough, 'year' : year, 'target' : target, 'predicted' : predicted, 'error' : error})
        self.forecast_df = pd.DataFrame(output)
        return self.forecast_df
   
    def plot_regression_line(self, ax : plt.axes, data: pd.DataFrame):
        sns.regplot(
            x = data['year'], 
            y = data['predicted'],
            ax = ax,
            scatter = False,
            ci = None,
            color = '#00BFFF',
            line_kws = {'alpha': 0.3, 'zorder': 0})
        
    def plot_error_bars(self, ax : plt.axes, data: pd.DataFrame):
        ax.errorbar(
            x = data['year'],
            y = data['target'],
            yerr = data['error'],
            alpha = 0.2,
            fmt = 'none',
            color = '#00BFFF', 
            zorder = 1,
            capsize = 3)
    
    def plot_forecast(self) -> sns.FacetGrid:
        if self.forecast_df is None:
            self.fit_predict()
        colours = ['#808080'] * len(self.INCLUDED_YEARS) + ['#00BFFF'] * len(self.FORECAST_YEARS)
        g = sns.relplot(kind = 'scatter', 
                        data = self.forecast_df, 
                        x = 'year', 
                        y = 'target', 
                        col = 'borough', 
                        col_wrap = 8,
                        height = 2, 
                        aspect = 1.4,
                        s = 50,
                        palette = 'RdYlGn',
                        hue = 'target',
                        legend = False,
                        edgecolors = colours, 
                        linewidth = 1,
                        zorder = 2)
        for borough, ax in g.axes_dict.items():
            borough_df = self.forecast_df[self.forecast_df['borough'] == borough]
            self.plot_regression_line(ax = ax, data = borough_df)
            self.plot_error_bars(ax = ax, data = borough_df[borough_df['year'].isin(self.FORECAST_YEARS)])
        g.set_titles('{col_name}', weight = 'bold')
        g.set(xlabel = 'Year', ylabel = f'Avg {self.target_col.upper()}')
        g.set_xticklabels([])
        if self.ADJUST_COVID:
            plt.savefig(f'src/stream_1/figures/Bayesian Borough Forecast - {self.target_col.upper()} (COVID Adjusted)', bbox_inches = 'tight')
        else:
            plt.savefig(f'src/stream_1/figures/Bayesian Borough Forecast - {self.target_col.upper()}', bbox_inches = 'tight')
        plt.show()
        return g

    def validate(self) -> pd.DataFrame:
        TRAIN_YEARS = [year for year in self.INCLUDED_YEARS if year not in self.TEST_YEARS] 
        borough_df = self.sort_and_group_boroughs()
        errors = []
        for borough in borough_df['LA_Name'].unique():
            X_train = (np.array(TRAIN_YEARS) - 2017).reshape(-1, 1)
            X_test = (np.array(self.TEST_YEARS) - 2017).reshape(-1, 1)
            Y_train = borough_df[(borough_df['LA_Name'] == borough) & (borough_df['year'].isin(TRAIN_YEARS))][self.target_col].values
            Y_test = borough_df[(borough_df['LA_Name'] == borough) & (borough_df['year'].isin(self.TEST_YEARS))][self.target_col].values
            model = BayesianRidge()
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            for year, actual, pred in zip(self.TEST_YEARS, Y_test, Y_pred):
                errors.append({'borough': borough, 'year' : year, 'error' : abs(actual - pred)})
        errors_df = pd.DataFrame(errors)
        return errors_df

    def plot_validation(self, errors_df : pd.DataFrame):
        _, ax = plt.subplots(figsize = (12, 6))
        sns.barplot(data = errors_df, x = 'borough', y = 'error', hue = 'year', palette = 'coolwarm_r', alpha = 0.75)
        ax.axhline(y = errors_df[errors_df['year'] == 2022]['error'].mean(), color = '#FF3131', linestyle = '--', zorder = 3, label = f'2022 Average Error ({errors_df[errors_df["year"] == 2022]["error"].mean():.1f})')
        ax.axhline(y = errors_df[errors_df['year'] == 2023]['error'].mean(), color = '#6495ED', linestyle = '--', zorder = 3, label = f'2023 Average Error ({errors_df[errors_df["year"] == 2023]["error"].mean():.1f})')
        ax.set_ylabel('Mean Absolute Error (2022/2023)', fontweight = 'bold', fontsize = 12)
        ax.set_xlabel('Borough', fontweight = 'bold', fontsize = 12)
        ax.legend()
        plt.xticks(rotation = 45, ha = 'right')
        plt.tight_layout()
        plt.savefig(f'src/stream_1/figures/Bayesian Borough Mean Average Error - {self.target_col.upper()}', bbox_inches = 'tight')
        plt.show()
    
if __name__ == "__main__":
    forecaster = BayesianBoroughForecaster()
    forecaster.fit_predict()
    forecaster.plot_forecast()
    forecaster.plot_validation(errors_df = forecaster.validate())

        