import numpy as np
from scipy.stats import norm


class GaussianMixture:

    def __init__(self, num_states, **param_inits):
        self._k = num_states
        self._mu, self._sigma, self_priors = param_inits

    def fit(self, X):
        self._X = X
        self._m, self._n = X.shape
        self._alpha = np.empty(shape=(self._m, self._k))

    def predict(self):
        pass
