from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score

class Logistic:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.f1 = None
        self.roc_auc = None

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def get_model(self):
        return self.model
    
    def get_preds(self, X_test, Y_test, save_metric : bool):
        preds = self.model.predict_proba(X_test)[:, 1]
        if save_metric:
            self.f1 = f1_score(Y_test, preds)
            self.roc_auc = roc_auc_score(Y_test, preds)
        return preds
    def get_roc_auc(self):
        return self.roc_auc
    def get_f1(self):
        return self.f1

    
