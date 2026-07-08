from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import pandas as pd

class ExponentialSmoothingModels:
    def __init__(self, model):
        self.models = ['holt', 'simple_exp', 'exp']
        if model not in self.models:
            raise KeyError(f'{model} not in {self.models} ')
        self.model = model
        self.forecast = {
            'holt' : self.fit_forecast_holt,
            'simple_exp' : self.fit_forecast_simple_exp,
            'exp' : self.fit_forecast_exp
        }
    def fit_forecast_holt(self, data : pd.Series, steps : int):
        holt = Holt(data, damped_trend = True)
        holt_fit = holt.fit()
        return holt_fit.forecast(steps)
    def fit_forecast_simple_exp(self, data : pd.Series, steps : int):
        simple = SimpleExpSmoothing(data)
        simple_fit = simple.fit()
        return simple_fit.forecast(steps)
    def fit_forecast_exp(self, data : pd.Series, steps : int):
        exp = ExponentialSmoothing(data)
        exp_fit = exp.fit()
        return exp_fit.forecast(steps)
    def estimate(self, data : pd.Series, steps : int):
        return self.forecast[self.model](data, steps)