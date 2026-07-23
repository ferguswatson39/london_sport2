from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
import optuna

class RFClassifier:
    """ Random Forest Classifier class
    Adapted from: https://medium.com/cloudvillains/random-forest-with-grid-search-b739fb0da311
    
    """
    def __init__(self):
        self.hyperparams = {
            # Lower max depth means less overfitting for smaller datasets
            'max_depth' : [3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, None],
            'n_estimators' : [100, 150, 200, 250, 300, 350, 400, 450, 500],
            'max_features' : ['sqrt', 'log2', None],
            'min_samples_leaf' : [1, 2 ,3, 4, 5, 6, 7, 8, 9, 10]
        }
        self.best_params = None
        self.model = RandomForestClassifier(random_state = 42)
        self.grid = GridSearchCV(self.model, self.hyperparams, cv=3, scoring='f1', verbose=3, n_jobs=-1)

    def fit(self, X_train, Y_train):
        best_params = self.search()
        self.model.set_params(**best_params)
        self.model.fit(X_train, Y_train)
        return self.get_model()
    def search(self):
        print('Tuning Random Forest Hyperparams.....')
        self.grid.fit(self.X_train, self.Y_train)
        self.best_params = self.grid.best_params_
        return self.get_best_params()
    def get_model(self):
        return self.model
    def get_score(self):
        return self.model.score(self.X_test, self.Y_test)
    def get_class_preds(self):
        model = self.get_model()
        preds = model.predict(self.X_test)
        return preds
    def get_probability_preds(self, X_test):
        """ Returns probabiltiies for both classes"""
        model = self.get_model()
        preds = model.predict_proba(X_test)
        return preds
    def get_classification_report(self):
        preds = self.get_class_preds() 
        return classification_report(self.Y_test, preds)
    def get_feature_importance(self, X_cols : list[str]) -> pd.DataFrame:
        features = pd.DataFrame(self.model.feature_importances_, index=X_cols) 
        return features
    def get_best_params(self):
        return self.best_params
    def get_proba(self, X_test):
        model = self.get_model()
        preds = model.predict_proba(X_test)
        return preds[:,1]
    

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

    def objective(self, trial):
        hyperparameters = {
            'max_depth' : trial.suggest_int('max_depth', 3, 30),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 20)            
        }
        model = RandomForestClassifier(**hyperparameters, random_state = 42)
        cv_score = cross_val_score(model, self.X_train, self.Y_train, cv=5, scoring = 'roc_auc', n_jobs = -1)
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
        self.model.fit(X_train, Y_train)
    
    def get_preds(self, X_test, Y_test):
        preds = self.model.predict_proba(X_test)[:, 1]
        self.f1 = f1_score(Y_test, preds)
        self.roc_auc = roc_auc_score(Y_test, preds)
        return preds
    
    def get_model(self):
        return self.model