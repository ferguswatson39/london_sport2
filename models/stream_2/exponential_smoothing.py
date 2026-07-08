from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import pandas as pd

class ExponentialSmoothingModels:
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