from sklearn.base import BaseEstimator

class ResampledEnsemble(BaseEstimator):

    def __init__(self, n_rules=20):
        
        self._estimator_type = "classifier"
        self.n_rules = n_rules