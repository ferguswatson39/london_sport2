from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
import pickle

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
        self.accuracy = None
        self.confusion = None
        self.study = None
        self.name = self.__class__.__name__
        self.save_path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' 
        self.scaler = None

    def objective(self, trial):
        model = RandomForestClassifier(
            n_estimators = trial.suggest_int('n_estimators', 50, 150),
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10),
            random_state = 42,
            class_weight='balanced'  
        )
        k_fold = StratifiedKFold(n_splits = 5, shuffle =True, random_state = trial.number)
        cv_score = cross_val_score(model, self.X_train, self.Y_train, cv=k_fold, scoring = 'f1_macro')
        return cv_score.mean()
    
    def run_study(self):
        study = optuna.create_study(direction = 'maximize', sampler=TPESampler(seed=42))
        study.optimize(self.objective, n_trials = 25)
        self.hyperparams = study.best_params
        self.model = RandomForestClassifier(**self.hyperparams, random_state = 42, class_weight='balanced')
        return study
    
    def fit(self, X_train, Y_train, scaler):
        self.X_train, self.Y_train = X_train, Y_train
        print(f'Starting to run study for {self.name}....')
        self.study = self.run_study()
        print(f'Optimal hyperparameters found for {self.name}.')
        print(f'Fitting {self.name}...')
        self.model.fit(X_train, Y_train)
        self.scaler = scaler

    
    def get_preds(self, X_test, Y_test, save_metric : bool):
        preds_prob = self.model.predict_proba(X_test)
        preds_class = self.model.predict(X_test)
        if save_metric:
            self.f1 = f1_score(Y_test, preds_class, average='macro')
            # Use 'ovo' to adjust for class imbalances
            self.roc_auc_score = roc_auc_score(Y_test, preds_prob[:, 1])
            self.accuracy = accuracy_score(Y_test, preds_class)
            self.confusion = confusion_matrix(Y_test, preds_class)
        return preds_prob
    
    def get_model(self):
        return self.model
    
    def get_roc_auc(self):
        return self.roc_auc_score
    def get_f1(self):
        return self.f1
    def get_accuracy(self):
        return self.accuracy
    def get_confusion(self):
        return self.confusion
    
    def save_class(self, run_num : int):
        filename = f'{run_num}_{self.name}.sav'
        path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' / 'classifiers' / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.name} saved to: {path}')

    def get_scaler(self):
        return self.scaler