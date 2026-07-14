from prepare_boroughs import prepare_borough_data
from sklearn.linear_model import BayesianRidge
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bayesian_ridge_forecast_borough(included_years : list, forecast_years: list, target_col: str) -> pd.DataFrame:
    borough_df = prepare_borough_data()
    borough_df = borough_df[borough_df['year'].isin(included_years)]
    output = []
    for borough in borough_df['LA_Name'].unique():
        X_train = (np.array(included_years) - 2017).reshape(-1, 1)
        X_forecast = (np.array(forecast_years) - 2017).reshape(-1, 1)
        Y_train = borough_df[borough_df['LA_Name'] == borough][target_col].values
        model = BayesianRidge()
        model.fit(X_train, Y_train)
        Y_forecast, E_forecast = model.predict(X_forecast, return_std = True)    
        years_full = included_years + forecast_years
        participation_full = np.concat((Y_train, Y_forecast))
        predicted_full = model.predict(np.concat((X_train, X_forecast))) 
        error_full = np.concat((np.zeros(len(Y_train)), E_forecast))
        for year, participation, error, predicted in zip(years_full, participation_full, error_full, predicted_full):
            output.append({'borough': borough, 'year' : year, 'participation' : participation, 'error' : error, 'predicted' : predicted})
    return pd.DataFrame(output)

# Forecast Visualisation
if __name__ == "__main__":
    INCLUDED_YEARS =  [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    FORECAST_YEARS = [2024, 2025, 2026, 2027]
    full_df = bayesian_ridge_forecast_borough(INCLUDED_YEARS, FORECAST_YEARS, target_col = 'MEMS7_ALL')
    colours = ['#808080'] * len(INCLUDED_YEARS) + ['#00BFFF'] * len(FORECAST_YEARS)
    g = sns.relplot(kind = 'scatter', 
                    data = full_df, 
                    x = 'year', 
                    y = 'participation', 
                    col = 'borough', 
                    col_wrap = 8,
                    height = 2, 
                    aspect = 1.4,
                    s = 50,
                    palette = 'RdYlGn',
                    hue = 'participation',
                    legend = False,
                    edgecolors = colours, 
                    linewidth = 1,
                    zorder = 2)
    
    for borough, ax in g.axes_dict.items():
        borough_df = full_df[full_df['borough'] == borough]
        # Add regression line
        sns.regplot(data = borough_df, 
                    x = 'year', 
                    y = 'predicted',
                    ax = ax,
                    scatter = False,
                    ci = None,
                    color = '#00BFFF',
                    line_kws = {'alpha': 0.3, 'zorder': 0})
        
        # Add erorr bars to forecast datapoints
        forecast_df = borough_df[borough_df['year'].isin(FORECAST_YEARS)]
        ax.errorbar(x = forecast_df['year'],
                    y = forecast_df['participation'],
                    yerr = forecast_df['error'] * 2,
                    alpha = 0.2,
                    fmt='none',
                    color = '#00BFFF', 
                    zorder = 1,
                    capsize = 3)
        
    g.set_titles('{col_name}', weight = 'bold')
    g.set(xlabel = 'Year', ylabel = 'Avg Sport Participation')
    g.set_xticklabels([])
    g.set_yticklabels([])
    plt.savefig('src/stream_1/figures/Borough Forecast with Uncertainy', bbox_inches = 'tight')
    plt.show()
