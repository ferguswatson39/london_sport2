import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


class Arima:
    def __init__(self, data):
        self.data = data
    
    def fit(self, data : pd.Series, pdq: tuple):
        model = ARIMA(data, order = pdq)
        return model.fit()

