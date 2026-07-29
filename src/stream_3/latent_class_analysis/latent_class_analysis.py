from stepmix.stepmix import StepMix
import pandas as pd
import numpy as np
import pickle

class LatentClassAnalysis:
    """
    Class for fitting and evaluating latent class models.
    """
    def __init__(
            self,
            n_components,
            measurement="categorical",
            random_state=42,
            n_init=10,
            max_iter=1000,
            abs_tol=1e-5
        ):
        self.model = StepMix(
            n_components=n_components,
            measurement=measurement,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            abs_tol=abs_tol
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X):
        model = self.model

        posterior = self.predict_proba(X)
        predicted = self.predict(X)

        proportions = posterior.mean(axis=0)
        assignment = posterior.max(axis=1)

        loglik = model.score(X) * len(X)

        return {"Classes": model.n_components,
               "LogLik": loglik,
               "AIC": model.aic(X),
               "BIC": model.bic(X),
               "Converged": model.converged_,
               "Iterations": model.n_iter_,
               "MeanAssignment": assignment.mean(),
               "MedianAssignment": np.median(assignment),
               "MinClass": proportions.min(),
               "MaxClass": proportions.max(),
               "ClassProportions": pd.Series(proportions),
               "ObservedClassSizes": pd.Series(predicted).value_counts(normalize=True).sort_index(),
               "EffectiveClasses": (proportions > 0.01).sum(),
               "PosteriorAssignment": pd.Series(assignment).describe()}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)

    lca = LatentClassAnalysis(n_components=model.n_components)
    lca.model = model

    return lca