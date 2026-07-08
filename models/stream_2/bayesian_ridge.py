from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np

class Bayesian:
    def fit_forecast_bayesian_ridge(self, data : np.ndarray, steps : int):
        model = BayesianRidge()
        data_len = np.arange(len(data))
        pred_len = np.arange(len(data), len(data) + steps)
        X_train = data_len.reshape(-1,1)
        y_train = data
        model.fit(X_train, y_train)
        y_pred, error = model.predict(pred_len.reshape(-1,1), return_std  = True)
        return y_pred, error

