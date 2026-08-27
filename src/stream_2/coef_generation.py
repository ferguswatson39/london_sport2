from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
from src.loading_data.load_data import get_coefficient_data
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.loading_data.data_catalogue import DataCatalogue

"""
File contians the CoefGeneration class which was used as part of EDA.
Class has multiple methods enabling it to fit OLS or Logistic model.
"""

LABEL_MAP = {'WorkStat8_2' : 'Work Status: Unemployed', 'WorkStat8_4' : 'Work Status: Domestic',
             'IMD10_9' : 'Deprivation: 9 (Low)', 'IMD10_8' : 'Deprivation: 8', 'IMD10_7' : 'Deprivation: 7',
             'IMD10_6' : 'Deprivation: 6', 'IMD10_5' : 'Deprivation: 5', 'IMD10_4' : 'Deprivation: 4',
             'IMD10_3' : 'Deprivation: 3', 'IMD10_2' : 'Deprivation: 2', 'IMD10_1' : 'Deprivation: 1 (High)'}

class CoefGeneration:
    def __init__(self, model: str, df : pd.DataFrame):
        self.df = df
        self.model = model
        self.model_catalogue = {
            'logistic' : {
                'fit_model' : self.fit_logistic,
                'target' : 'active',
                'save_path' : Path(ROOT / 'data' / 'logistic_coefficients'),
                'file_name' : 'logistic_coef_results.csv',
                'plot_values' : ['odds_ratios', 'pvalues', 'confidence_lower', 'confidence_upper', 'std_error'],
                'plot_title' : 'Odds ratio of participating in >150 minutes of PA per week, 2016-2023',
                'main_var' : 'odds_ratios', 
                'fig_save_path' : Path(ROOT / 'figures' / 'logistic_coef_plot.png'),
                'fig_save_path_trends' : Path(ROOT / 'figures' / 'logistic_coef_trend_plot.png'),
                'forecast_file_name' : 'logistic_forecast_results.csv',
                'forecast_plot_path' : Path(ROOT / 'figures' / 'logistic_forecasts')},

            'ols' : {
                'fit_model' : self.fit_ols,
                'target' : 'LOG_MEMS7_ALL',
                'save_path' : Path(ROOT / 'data' / 'ols_coefficients'),
                'file_name' : 'ols_coef_results.csv',
                'plot_values' : ['coef_as_percent', 'pvalues', 'confidence_lower', 'confidence_upper', 'std_error'],
                'plot_title' : 'Percentage change in Activity (MEMS), per unit change in predictor (2016-2023)',
                'main_var' : 'coef_as_percent',
                'fig_save_path' : Path(ROOT / 'figures' / 'ols_coef_plot.png'),
                'fig_save_path_trends' : Path(ROOT / 'figures' / 'ols_coef_trend_plot.png'),
                'forecast_file_name' : 'ols_forecast_results.csv',
                'forecast_plot_path' : Path(ROOT / 'figures' / 'ols_forecasts')}
        }

        if self.model not in ['logistic', 'ols']:
            raise KeyError(f"{self.model} not in ['logistic', 'ols'] ")
    def fit_logistic(self, Y, X):
        return sm.Logit(Y, X).fit()
    def fit_ols(self, Y, X):
        return sm.OLS(Y, X).fit()
    
    def check_empty_dummies(self, X : pd.DataFrame, year : str) -> list[str]:
        """ Checks if any vars are completely 0 - want to prevent dummies being used that have zero coef"""
        no_obs = []
        num_zeroes = (X[X== 0].count()/len(X)).sort_values(ascending=False) 
        for idx, value in enumerate(num_zeroes):
            if value == 1.0:
                no_obs.append(num_zeroes.index[idx])
        if len(no_obs) > 0:
            print(f'Year: {year}. No observations detected for:')
            print(no_obs)
            return no_obs
        else:
            print(f'Year: {year}. All columns have observations')
            return []

    def build_csv(self, model, output, year) -> pd.DataFrame:
        if model == 'logistic':
            return pd.DataFrame({
                            'feature_names' : output.params.index,
                            'odds_ratios' : np.exp(output.params.values),
                            'pvalues' : output.pvalues.values,
                            'confidence_lower' : np.exp(output.conf_int()[0].values),
                            'confidence_upper' : np.exp(output.conf_int()[1].values),
                            'std_error' : output.bse.values,
                            'pseudo_rsquared' : output.prsquared,
                            'year' : year })
        elif model == 'ols' : 
            return pd.DataFrame({
                        'feature_names' : output.params.index,
                        'coef' : output.params.values,
                        'coef_as_percent' : ((np.exp(output.params.values) - 1) * 100),
                        'pvalues' : output.pvalues.values,
                        'confidence_lower' : output.conf_int()[0].values,
                        'confidence_upper' : output.conf_int()[1].values,
                        'std_error' : output.bse.values,
                        'rsquared' : output.rsquared,
                        'adj_rsquared' : output.rsquared_adj,
                        'year' : year })

    def generate_coefs(self):
        dc = DataCatalogue()
        discrete_variables = dc.get_to_be_encoded_vars()
        encoder = OneHotEncoder(drop='first', handle_unknown = 'ignore', sparse_output = False).set_output(transform = 'pandas')
        encoded = encoder.fit_transform(self.df[discrete_variables])
        df_encoded = pd.concat([self.df, encoded], axis = 1).drop(columns = discrete_variables)
        all_years = []
        for year in sorted(df_encoded['year'].unique()):
            df_encoded_sy = df_encoded[df_encoded['year'] == year]
            df_encoded_sy = df_encoded_sy.drop(columns=['year'])
            Y = df_encoded_sy[self.model_catalogue[self.model]['target']]
            X = df_encoded_sy.drop(columns = dc.get_target_vars() + ['serial'], errors = 'ignore')
            X = sm.add_constant(X)
            continuous_cols = dc.get_continuous_vars(with_targets = False)
            if len(continuous_cols) > 0:
                scaler = StandardScaler()
                X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
            no_obs = self.check_empty_dummies(X, year)
            if len(no_obs) > 0:
                X = X.drop(columns = no_obs)
            # print(f'Columns into Model:\n>>> Y = {Y.name}\n>>> X = {X.columns}')
            output = self.model_catalogue[self.model]['fit_model'](Y, X)
            all_years.append(self.build_csv(self.model, output, year))
        all = pd.concat(all_years)
        print(f'Coefs generated successfully for: {self.model}')
        return all
    
    def save_results(self, results):
        results.to_csv(self.model_catalogue[self.model]['save_path'] / self.model_catalogue[self.model]['file_name'], index=False )
        print(f"Results for {self.model} saved successfully to: {self.model_catalogue[self.model]['save_path'] / self.model_catalogue[self.model]['file_name']}")

    def generate_forest_plot(self):
        df = pd.read_csv(self.model_catalogue[self.model]['save_path'] / self.model_catalogue[self.model]['file_name'])
        df = df[df['feature_names'] != 'const']
        df = df[df['feature_names'].isin(LABEL_MAP.keys())].copy()
        df['feature_names'] = df['feature_names'].map(LABEL_MAP)
        pivoted = df.pivot(
            index='feature_names',
            columns='year',
            values=self.model_catalogue[self.model]['plot_values']
        )
        if self.model == 'ols':
            line = 0
        elif self.model == 'logistic':
            line = 1
        fig, ax = plt.subplots(figsize=(10, 4))
        num_years = len(pivoted.columns.levels[1])
        ax.set_prop_cycle(color = plt.cm.YlOrRd(np.linspace(0,1,num_years)))
        for _, year in enumerate(sorted(pivoted.columns.levels[1])):
            ax.errorbar(x=pivoted[self.model_catalogue[self.model]['main_var']][year].values, y =pivoted.index, marker='o', linestyle='None', label=year)
            ax.legend(loc = 'upper right', labelspacing = 0.25)
            ax.axvline(x=line, linestyle='--')
        fig.savefig(self.model_catalogue[self.model]['fig_save_path'], bbox_inches = 'tight')
        print(f"Coef forest plot generated successfully for {self.model}.\nFigure saved to {self.model_catalogue[self.model]['fig_save_path']}")



if __name__ == '__main__':
    df = get_coefficient_data()
    OLS = CoefGeneration('ols', df)
    OLS.save_results(OLS.generate_coefs())
    LOGISTIC = CoefGeneration('logistic', df)
    LOGISTIC.save_results(LOGISTIC.generate_coefs())
    OLS.generate_forest_plot()
    LOGISTIC.generate_forest_plot()