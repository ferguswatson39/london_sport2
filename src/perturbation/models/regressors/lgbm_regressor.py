from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import optuna
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.append(str(ROOT))
import pickle


class LightGBMRegressor:
    def __init__(self):
        self.hyperparams = None
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.mse = None
        self.rmse = None
        self.study = None

    def objective(self, trial):
        hyperparameters = {
            'max_depth' : trial.suggest_int('max_depth', 3, 30),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate' : trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
            'num_leaves' : trial.suggest_int('num_leaves', 5, 50),
            'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_freq' : trial.suggest_int('bagging_freq', 1, 10),
            'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 10, 50)     
        }
        model = LGBMRegressor(**hyperparameters, random_state = 42)
        cv_score = cross_val_score(model, self.X_train, self.Y_train, cv=5, scoring = 'neg_mean_squared_error')
        return cv_score.mean()
    
    def run_study(self):
        # direction = maximise as neg_mean_squared_error is score
        study = optuna.create_study(direction = 'maximize')
        study.optimize(self.objective, n_trials = 100)
        self.hyperparams = study.best_params
        self.model = LGBMRegressor(**self.hyperparams, random_state = 42)
        return study
    
    def fit(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train
        print(f'Starting to run study....')
        self.study = self.run_study()
        print(f'Finished running study. Optimal hyperparameters found.')
        print(f'Fitting {LightGBMRegressor.__name__}...')
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
    def save_class(self):
        filename = f'{self.__class__.__name__}.sav'
        path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.__class__.__name__} saved to: {path}')


"""
Apparently, searching through 'n_estimators' : trial.suggest_int('n_estimators', 100, 1000) when i use /
        self.model.fit(X_train, Y_train, 
                       eval_set = [(X_val, Y_val)],
                       callbacks = [early_stopping(stopping_rounds=100), log_evaluation(period=100)]
                    )
is pointless as you have early stopping which prevents overfitting.
so i should either search in the n_estimators space or set a high n_estimators value and use early stopping

"""

