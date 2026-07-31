from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
import optuna
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
import pickle

class LightGBMClassifier:
    def __init__(self):
        self.hyperparams = None
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.f1 = None
        self.roc_auc_score = None
        self.study = None
        self.name = LightGBMClassifier.__name__
        self.save_path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' 

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
        model = LGBMClassifier(**hyperparameters, random_state = 42, verbose=-1)
        # Scoring here is roc_auc but maybe i should try_ f1
        cv_score = cross_val_score(model, self.X_train, self.Y_train, cv=5, scoring = 'roc_auc')
        return cv_score.mean()
    
    def run_study(self):
        # direction = maximise as neg_mean_squared_error is score
        study = optuna.create_study(direction = 'maximize')
        study.optimize(self.objective, n_trials = 100)
        self.hyperparams = study.best_params
        self.model = LGBMClassifier(**self.hyperparams, random_state = 42, verbose = -1)
        return study
    
    def fit(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train
        print(f'Starting to run study....')
        self.study = self.run_study()
        print(f'Finished running study. Optimal hyperparameters found.')
        print(f'Fitting {LightGBMClassifier.__name__}...')
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
    
    def save_class(self):
        filename = f'{self.__class__.__name__}.sav'
        path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.__class__.__name__} saved to: {path}')

