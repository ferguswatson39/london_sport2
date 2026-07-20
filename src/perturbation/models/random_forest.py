from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV

class RandomForest:
    """ Random Forest Classifier class
    Adapted from: https://medium.com/cloudvillains/random-forest-with-grid-search-b739fb0da311
    
    """
    def __init__(self, X_train, X_test, Y_train, Y_test, X_cols):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.X_cols = X_cols
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

    def fit_rf(self):
        best_params = self.search()
        self.model.set_params(**best_params)
        self.model.fit(self.X_train, self.Y_train)
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
    def get_probability_preds(self, X):
        """ Returns probabiltiies for both classes"""
        model = self.get_model()
        preds = model.predict_proba(X)
        return preds
    def get_classification_report(self):
        preds = self.get_class_preds() 
        return classification_report(self.Y_test, preds)
    def get_feature_importance(self) -> pd.DataFrame:
        features = pd.DataFrame(self.model.feature_importances_, index=self.X_cols) 
        return features
    def get_best_params(self):
        return self.best_params