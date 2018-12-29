import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from data_generators import generate_data
from scipy.stats import norm
from gaussian_mixture import GaussianMixture

k = 2
mu = [1, 4.5]
sigma = [1, 1]
priors = [.5, .5]
num_points = 300

print "\nTrue priors: ", priors
print "True mus: ", mu
print "True sigmas: ", sigma

components = [norm(mu[idx], sigma[idx]) for idx in range(k)]
data, pdfs = generate_data(num_points, 'gaussian', num_states=k, mu=mu, sigma=sigma, priors=priors)

initial_mu = np.linspace(start=-5, stop=5, num=k)
initial_sigma = np.ones(k)
initial_priors = np.ones(k, dtype=np.float)
initial_priors /= initial_priors.sum()

em = GaussianMixture(k, mu=initial_mu, sigma=initial_sigma, priors=initial_priors)
em.fit(data)
pred_priors, (pred_mu, pred_sigma) = em.predict()
print "\nPredicted priors: ", pred_priors
print "Predicted mus: ", pred_mu
print "Predicted sigmas: ", pred_sigma

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