from prepare_boroughs import prepare_borough_data
from sklearn.linear_model import LinearRegression
from scipy.special import logit, expit
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class LogisticBoroughForecaster:
    def __init__(self, 
                 INCLUDED_YEARS : List[int] = [2017, 2018, 2019, 2020, 2021, 2022, 2023],
                 FORECAST_YEARS : List[int] = [2024, 2025, 2026, 2027],
                 target_col: str = 'active',
                 ADJUST_COVID: bool = False,
                 BOROUGH_ORDER: List[str] = None,
                 BOROUGH_HIGHLIGHT: List[str] = None
    ):
        self.INCLUDED_YEARS = INCLUDED_YEARS
        self.FORECAST_YEARS = FORECAST_YEARS
        self.target_col = target_col
        self.ADJUST_COVID = ADJUST_COVID
        self.BOROUGH_ORDER = BOROUGH_ORDER
        self.BOROUGH_HIGHLIGHT = BOROUGH_HIGHLIGHT
        self.forecast_df : pd.DataFrame = None

    def fit_predict(self) -> pd.DataFrame:
        borough_df = prepare_borough_data(ADJUST_COVID = self.ADJUST_COVID, target_col = self.target_col)
        borough_df = borough_df[borough_df['year'].isin(self.INCLUDED_YEARS)]
        borough_df[self.target_col] = borough_df.groupby('LA_Name')[self.target_col].ffill()
        output = []
        for borough in borough_df['LA_Name'].unique():
            X_train = (np.array(self.INCLUDED_YEARS) - 2017).reshape(-1, 1)
            X_forecast = (np.array(self.FORECAST_YEARS) - 2017).reshape(-1, 1)
            X_full = np.concat((X_train, X_forecast))
            Y_train = borough_df[borough_df['LA_Name'] == borough][self.target_col].values
            MIN, MAX = Y_train.min() -  np.std(Y_train), Y_train.max() + np.std(Y_train)
            Y_train_scaled = (Y_train - MIN) / (MAX - MIN)
            Y_train_logit = logit(Y_train_scaled)
            model = LinearRegression()
            model.fit(X_train, Y_train_logit)
            Y_forecast = (MIN + (expit(model.predict(X_forecast)) * (MAX - MIN))) * 100
            Y_predicted = (MIN + (expit(model.predict(X_full)) * (MAX - MIN))) * 100
            Y_target = np.concat((Y_train * 100, Y_forecast))
            for year, target, predicted in zip(self.INCLUDED_YEARS + self.FORECAST_YEARS, Y_target, Y_predicted):
                output.append({'borough': borough, 'year' : year, 'target' : target, 'predicted' : predicted})
        self.forecast_df = pd.DataFrame(output)
        return self.forecast_df
    
    def plot_logistic_line(self, ax : plt.axes, data: pd.DataFrame):
        sns.lineplot(
            x = data['year'], 
            y = data['predicted'],
            ax = ax,
            color = '#00BFFF',
            alpha = 0.3, 
            zorder = 0)
    
    def plot_forecast(self, highlight : bool = False) -> sns.FacetGrid:
        if self.forecast_df is None:
            self.fit_predict()
        if highlight: boroughs = self.BOROUGH_HIGHLIGHT
        else: boroughs = self.BOROUGH_ORDER
        colours = ['#808080'] * len(self.INCLUDED_YEARS) + ['#00BFFF'] * len(self.FORECAST_YEARS)
        g = sns.relplot(kind = 'scatter', 
                        data = self.forecast_df, 
                        x = 'year', 
                        y = 'target', 
                        col = 'borough', 
                        col_wrap = 4,
                        col_order = boroughs,
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
            self.plot_logistic_line(ax = ax, data = borough_df)
        g.set_titles('{col_name}', weight = 'bold')
        g.set(xlabel = 'Year', ylabel = f'Volunteering %')
        tick_years = [2017, 2019, 2021, 2023, 2025, 2027]
        g.set(xlim = (2016.5, 2027.5), xticks = tick_years)
        if highlight: 
            last_ax = g.axes_dict[boroughs[-1]]
            last_ax.text(s = '(+28 boroughs)', x = 2029, y = 17, fontsize = 12, ha = 'left', va = 'center', color = '#999999', weight = 'bold')
        if self.ADJUST_COVID:
            plt.savefig(f'figures/borough/Bounded Borough Forecast - {self.target_col.upper()} - Highlight {highlight} (COVID Adjusted)', bbox_inches = 'tight', dpi = 400)
        else:
            plt.savefig(f'figures/borough/Bounded Borough Forecast - {self.target_col.upper()} - Highlight {highlight}', bbox_inches = 'tight', dpi = 400)
        plt.show()
        return g
    
if __name__ == "__main__":
    borough_highlight = ['Barking and Dagenham', 'Barnet', 'Hackney' , 'Wandsworth']
    borough_order = [
        'Barking and Dagenham', 'Barnet', 'Hackney', 'Wandsworth', 
        'Bexley', 'Brent', 'Bromley', 'Camden',
        'Croydon', 'Ealing', 'Enfield', 'Greenwich', 
        'Tower Hamlets', 'Hammersmith and Fulham', 'Haringey', 'Harrow',
        'Havering', 'Hillingdon', 'Hounslow', 'Richmond', 
        'Kensington and Chelsea', 'Kingston', 'Lambeth', 'Lewisham',
        'Merton', 'Newham', 'Redbridge', 'Southwark', 
        'Sutton', 'Islington', 'Waltham Forest', 'Westminster'
    ]
    forecaster = LogisticBoroughForecaster(BOROUGH_ORDER = borough_order, BOROUGH_HIGHLIGHT = borough_highlight, target_col='VolAny')
    forecaster.fit_predict()
    forecaster.plot_forecast(highlight = False)