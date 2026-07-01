from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
from src.loading_data.load_data import get_data
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.loading_data.data_catalogue import DataCatalogue

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
                'fig_save_path' : Path(ROOT / 'figures' / 'logistic_coef_plot.png')},
            'ols' : {
                'fit_model' : self.fit_ols,
                'target' : 'LOG_MEMS7_ALL',
                'save_path' : Path(ROOT / 'data' / 'ols_coefficients'),
                'file_name' : 'ols_coef_results.csv',
                'plot_values' : ['coef_as_percent', 'pvalues', 'confidence_lower', 'confidence_upper', 'std_error'],
                'plot_title' : 'Percentage change in MEMS, for a one unit change in the predictor, 2016-2023',
                'main_var' : 'coef_as_percent',
                'fig_save_path' : Path(ROOT / 'figures' / 'ols_coef_plot.png')}
        }
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
            print(f'Year: {year}. No observations detected!:')
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
            X = df_encoded_sy.drop(columns=dc.get_target_vars())
            X = sm.add_constant(X)
            scaler = StandardScaler()
            continuous_cols = dc.get_continuous_vars(with_targets = False)
            X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
            no_obs = self.check_empty_dummies(X, year)
            if len(no_obs) > 0:
                X = X.drop(columns = no_obs)
            print(f'Columns into Model:\n>>> Y = {Y.name}\n>>> X = {X.columns}')
            output = self.model_catalogue[self.model]['fit_model'](Y, X)
            all_years.append(self.build_csv(self.model, output, year))
        all = pd.concat(all_years)
        return all
    
    def save_results(self, results):
        results.to_csv(self.model_catalogue[self.model]['save_path'] / self.model_catalogue[self.model]['file_name'], index=False )
        print(f"Results Saved to: {self.model_catalogue[self.model]['save_path'] / self.model_catalogue[self.model]['file_name']}")

    def generate_plot(self):
        df = pd.read_csv(self.model_catalogue[self.model]['save_path'] / self.model_catalogue[self.model]['file_name'])
        df = df[df['feature_names'] != 'const']
        pivoted = df.pivot(
            index='feature_names',
            columns='year',
            values=self.model_catalogue[self.model]['plot_values']
        )
        if self.model == 'ols':
            line = 0
        elif self.model == 'logistic':
            line = 1
        fig, ax = plt.subplots(figsize=(10,15))
        num_years = len(pivoted.columns.levels[1])
        ax.set_prop_cycle(color = plt.cm.YlOrRd(np.linspace(0,1,num_years)))
        for idx, year in enumerate(sorted(pivoted.columns.levels[1])):
            ax.errorbar(x=pivoted[self.model_catalogue[self.model]['main_var']][year].values, y =pivoted.index, marker='o', linestyle='None', label=year)
            ax.legend()
            ax.set_title(self.model_catalogue[self.model]['plot_title'])
            ax.axvline(x=line, linestyle='--')
        fig.savefig(self.model_catalogue[self.model]['fig_save_path'])
        print(f"Done. Figure saved to {self.model_catalogue[self.model]['fig_save_path']}")
