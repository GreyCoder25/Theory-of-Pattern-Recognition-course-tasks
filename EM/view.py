import matplotlib.pyplot as plt
import numpy as np
from data_generators import generate_data
from scipy.stats import norm

k = 2
mu = [-1, 1]
sigma = [1, 1]
priors = [.5, .5]
num_points = 100
components = [norm(mu[idx], sigma[idx]) for idx in range(k)]
data, y = generate_data(num_points, 'gaussian', num_states=2, mu=mu, sigma=sigma, priors=priors)
# plt.scatter(data, np.zeros_like(data), s=10, c=['blue'])
plt.scatter(data, y, s=10, c=['blue'])
x = np.linspace(-10, 10, 200)
for component in components:
    plt.plot(x, component.pdf(x), color='blue')
plt.xlim(-5, 5)
plt.ylim(-.1, .5)
plt.show()