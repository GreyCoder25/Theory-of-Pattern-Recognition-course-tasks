import numpy as np
from scipy.stats import norm


class GaussianMixture:

    def __init__(self, num_states, **param_inits):
        self._k = num_states
        self._mu, self._sigma, self._priors = param_inits.values()
        self._epsilon = 0.001
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
        self._model = (self._alpha, (self._priors, (self._mu, self._sigma)))

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

            new_model = (alpha_new, (priors_new, (mu_new, sigma_new)))
            if self._compare_models(self._model, new_model):
                model_changed = False
                self._gaussians = [norm(mu, sigma) for (mu, sigma) in zip(self._mu, self._sigma)]
            else:
                self._alpha = alpha_new
                self._priors = priors_new
                self._mu = mu_new
                self._sigma = sigma_new
                self._model = new_model
        return self._priors, (self._mu, self._sigma)

    def compute_gaussian_mixture_pdf(self, x):
        res = 0
        for idx, gaussian in enumerate(self._gaussians):
            res += self._priors[idx] * self._gaussians[idx].pdf(x)
        return res

    def _compare_models(self, model1, model2):
        """Returns True if models are equal"""
        (alpha1, (priors1, (mu1, sigma1))) = model1
        (alpha2, (priors2, (mu2, sigma2))) = model2
        return (np.array_equal(alpha1, alpha2) and
                np.array_equal(priors1, priors2) and
                np.array_equal(mu1, mu2) and
                np.array_equal(sigma1, sigma2))

