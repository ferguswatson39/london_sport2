from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
import pickle

class Logistic:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.f1 = None
        self.roc_auc = None
        self.name = Logistic.__name__
        self.save_path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' 


    def fit(self, X_train, Y_train):
        print(f'Fitting {Logistic.__name__}...')
        self.model.fit(X_train, Y_train)

    def get_model(self):
        return self.model
    
    def get_preds(self, X_test, Y_test, save_metric : bool):
        preds_prob = self.model.predict_proba(X_test)[:, 1]
        preds_class = self.model.predict(X_test)
        if save_metric:
            self.f1 = f1_score(Y_test, preds_class)
            self.roc_auc = roc_auc_score(Y_test, preds_prob)
        return preds_prob
    def get_roc_auc(self):
        return self.roc_auc
    def get_f1(self):
        return self.f1
    def get_name(self):
        return self.name
    
    def save_class(self):
        filename = f'{self.__class__.__name__}.sav'
        path = ROOT / 'src' / 'perturbation' / 'models' / 'saved_models' / filename
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f'{self.__class__.__name__} saved to: {path}')
    