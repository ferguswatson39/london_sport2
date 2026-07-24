from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

class OLSRegressor:
    def __init__(self):
        self.model = LinearRegression()
        self.mse = None

    def fit(self, X_train, Y_train):
        print(f'Fitting {OLSRegressor.__name__}...')
        self.model.fit(X_train, Y_train)
    
    def get_model(self):
        return self.model
    
    def get_preds(self, X_test, Y_test, save_metric : bool):
        preds = self.model.predict(X_test)
        if save_metric:
            self.mse = mean_squared_error(Y_test, preds)
        return preds
    def get_mse(self):
        return self.mse