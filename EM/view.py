import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from data_generators import generate_data
from scipy.stats import norm
from gaussian_mixture import GaussianMixture

k = 3
mu = [-3, 0, 3]
sigma = [1, 1, 1]
priors = [.33, .33, .34]
num_points = 100
components = [norm(mu[idx], sigma[idx]) for idx in range(k)]
data, pdfs = generate_data(num_points, 'gaussian', num_states=k, mu=mu, sigma=sigma, priors=priors)

initial_mu = [-3, 0, 3]
initial_sigma = [1, 1, 1]
initial_priors = [.33, .33, .34]
em = GaussianMixture(k, mu=initial_mu, sigma=initial_sigma, priors=initial_priors)
em.fit(data)
pred_priors, (pred_mu, pred_sigma) = em.predict()
print("Predicted priors: ", pred_priors)
print("Predicted mus: ", pred_mu)
print("Predicted sigmas: ", pred_sigma)

# plt.scatter(data, np.zeros_like(data), s=10, c=['blue'])
plt.scatter(data, pdfs, s=10, c=['blue'])
x = np.linspace(-10, 10, 200)
for component in components:
    plt.plot(x, component.pdf(x), color='blue')
# plot gaussian mixture
plt.plot(x, [em.compute_gaussian_mixture_pdf(point) for point in x], color='red')
plt.xlim(-10, 10)
plt.ylim(-.1, .5)
plt.show()