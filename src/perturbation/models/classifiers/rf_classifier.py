from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
import optuna
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

class RFClassifier:
    """ Random Forest Classifier class
    Adapted from: https://medium.com/cloudvillains/random-forest-with-grid-search-b739fb0da311
    
    """
    def __init__(self):
        self.hyperparams = None
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.f1 = None
        self.roc_auc_score = None
        self.name = RFClassifier.__name__
        self.save_path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' 

    def objective(self, trial):
        hyperparameters = {
            'max_depth' : trial.suggest_int('max_depth', 3, 30),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 20)            
        }
        model = RandomForestClassifier(**hyperparameters, random_state = 42)
        cv_score = cross_val_score(model, self.X_train, self.Y_train, cv=5, scoring = 'roc_auc')
        return cv_score.mean()
    
    def run_study(self):
        study = optuna.create_study(direction = 'maximize')
        study.optimize(self.objective, n_trials = 100)
        self.hyperparams = study.best_params
        self.model = RandomForestClassifier(**self.hyperparams, random_state = 42)
        return study
    
    def fit(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train
        print(f'Starting to run study....')
        self.run_study()
        print(f'Finished running study. Optimal hyperparameters found.')
        print(f'Fitting {RFClassifier.__name__}...')
        self.model.fit(X_train, Y_train)

    
    def get_preds(self, X_test, Y_test, save_metric : bool):
        preds_prob = self.model.predict_proba(X_test)[:, 1]
        preds_class = self.model.predict(X_test)
        if save_metric:
            self.f1 = f1_score(Y_test, preds_class)
            self.roc_auc = roc_auc_score(Y_test, preds_prob)
        return preds_prob
    
    def get_model(self):
        return self.model
    
    def get_roc_auc(self):
        return self.roc_auc
    def get_f1(self):
        return self.f1
    