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
        return self.get_model()

    def get_class_preds(self, X_test):
        model = self.get_model()
        preds = model.predict(X_test)
        return preds
    def get_classification_report(self, Y_test):
        preds = self.get_class_preds() 
        return classification_report(Y_test, preds)
    def get_model(self):
        return self.model
    
    def get_preds(self, X_test, Y_test):
        preds = self.model.predict_proba(X_test)[:, 1]
        self.f1 = f1_score(Y_test, preds)
        self.roc_auc = roc_auc_score(Y_test, preds)
        return preds
    
