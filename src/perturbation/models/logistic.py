from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class Logistic:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.model = LogisticRegression(random_state=42)

    def fit_logistic(self):
        self.model.fit(self.X_train, self.Y_train)
        return self.get_model()

    def get_class_preds(self):
        model = self.get_model()
        preds = model.predict(self.X_test)
        return preds
    def get_classification_report(self):
        preds = self.get_class_preds() 
        return classification_report(self.Y_test, preds)
    def get_model(self):
        return self.model
    def get_preds(self, X):
        preds = self.model.predict_proba(X)
        return preds[: , 1]
    
