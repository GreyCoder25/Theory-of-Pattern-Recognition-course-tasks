import numpy as np
import numpy.random as rnd
from scipy.stats import norm


def generate_data(num_points=100, distr='gaussian', num_states=2, **parameters):
    if distr == 'gaussian':
        return _generate_gaussian(num_points, num_states, **parameters)


def _generate_gaussian(num_points=100, num_states=2, mu=(-1, 1), sigma=(1, 1), priors=(.5, .5)):
    weights = priors
    mixture_idx = rnd.choice(num_states, size=num_points, p=weights)
    data = np.array([rnd.normal(loc=mu[idx], scale=sigma[idx]) for idx in mixture_idx])
    y = np.array([norm(mu[cmpnt_idx], sigma[cmpnt_idx]).pdf(data[data_idx])
                  for data_idx, cmpnt_idx in enumerate(mixture_idx)])
    return data, y
