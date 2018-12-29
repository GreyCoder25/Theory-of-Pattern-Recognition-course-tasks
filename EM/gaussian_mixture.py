import numpy as np
from scipy.stats import norm


class GaussianMixture:

    def __init__(self, num_states, **param_inits):
        self._k = num_states
        self._mu, self._sigma, self._priors = param_inits['mu'], param_inits['sigma'], param_inits['priors']
        self._epsilon = 1e-10
        # casting array-like parameters to ndarrays
        self._mu = np.array(self._mu)
        self._sigma = np.array(self._sigma)
        self._priors = np.array(self._priors)
        self._gaussians = [norm(mu, sigma) for mu, sigma in zip(self._mu, self._sigma)]

    def fit(self, X):
        self._X = np.array(X)

        if len(X.shape) > 1:
            self._m, self._n = X.shape
        else:
            self._m = X.shape[0]
            self._n = 1
            self._X = self._X.reshape(self._m, self._n)
        self._alpha = np.ones((self._m, self._k))

    def predict(self):
        model_changed = True
        while model_changed:
            if (self._alpha < self._epsilon).sum() == 0:
                gaussians = [norm(mu, sigma) for (mu, sigma) in zip(self._mu, self._sigma)]
                f = np.array([gaussian.pdf(self._X) for gaussian in gaussians]).T[0]
                alpha_new = f * self._priors
                alpha_new = (alpha_new.T / alpha_new.sum(axis=1)).T
            priors_new = alpha_new.sum(axis=0) / self._m
            mu_new = (alpha_new.T.dot(self._X).T / alpha_new.sum(axis=0))[0]
            sigma_new = (alpha_new.T * ((self._X - mu_new)**2).T).T.sum(axis=0) / alpha_new.sum(axis=0)

            if self._compare_models(self._alpha, alpha_new):
                model_changed = False
                self._gaussians = [norm(mu, sigma) for (mu, sigma) in zip(self._mu, self._sigma)]
            else:
                self._alpha = alpha_new
                self._priors = priors_new
                self._mu = mu_new
                self._sigma = sigma_new
        return self._priors, (self._mu, self._sigma)

    def compute_gaussian_mixture_pdf(self, x):
        res = 0
        for idx, gaussian in enumerate(self._gaussians):
            res += self._priors[idx] * self._gaussians[idx].pdf(x)
        return res

    def _compare_models(self, alpha1, alpha2):
        """Returns True if models are equal"""
        return np.array_equal(alpha1, alpha2)

    def log_likelihood_of_mu(self, mu_array):
        mu_array = np.array(mu_array)
        gaussians = [norm(mu, sigma) for (mu, sigma) in zip(mu_array, self._sigma)]
        f = np.array([gaussian.pdf(self._X) for gaussian in gaussians]).T[0]
        log_likelihood = f.dot(self._priors).sum()
        return log_likelihood

