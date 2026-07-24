from sklearn.ensemble import RandomForestRegressor
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

# optuna hp optimisation adapted from: 
# https://medium.com/@sarahzouinina/a-deep-dive-into-lightgbm-how-to-choose-and-tune-parameters-7c584945842e

class RFRegressor:
    def __init__(self):
        self.hyperparams = None
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.mse = None
        self.rmse = None

    def objective(self, trial):
        hyperparameters = {
            'max_depth' : trial.suggest_int('max_depth', 3, 30),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 20)            
        }
        model = RandomForestRegressor(**hyperparameters, random_state = 42)
        cv_score = cross_val_score(model, self.X_train, self.Y_train, cv=5, scoring = 'neg_mean_squared_error')
        return cv_score.mean()
    
    def run_study(self):
        study = optuna.create_study(direction = 'maximize')
        study.optimize(self.objective, n_trials = 100)
        self.hyperparams = study.best_params
        self.model = RandomForestRegressor(**self.hyperparams, random_state = 42)
        return study
    
    def fit(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train
        print(f'Starting to run study....')
        self.run_study()
        print(f'Finished running study. Optimal hyperparameters found.')
        print(f'Fitting {RFRegressor.__name__}...')
        self.model.fit(X_train, Y_train)

    def get_preds(self, X_test, Y_test, save_metric : bool):
        preds = self.model.predict(X_test)
        if save_metric:
            self.mse = mean_squared_error(Y_test, preds)
            self.rmse = np.sqrt(self.get_mse())
        return preds

    def get_model(self):
        return self.model

    def get_mse(self):
        return self.mse
    def get_rmse(self):
        return self.rmse
    
    
