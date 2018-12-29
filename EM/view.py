import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from data_generators import generate_data
from scipy.stats import norm
from gaussian_mixture import GaussianMixture

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

X_MIN = -10
X_MAX = 10
Y_MIN = -.1
Y_MAX = .5

k = 2
mu = [-2, 4]
sigma = [1, 1]
priors = [.5, .5]
num_points = 200

print("\nTrue priors: ", priors)
print("True mus: ", mu)
print("True sigmas: ", sigma)

components = [norm(mu[idx], sigma[idx]) for idx in range(k)]
data, pdfs = generate_data(num_points, 'gaussian', num_states=k, mu=mu, sigma=sigma, priors=priors)

initial_mu = np.linspace(start=X_MIN, stop=X_MAX, num=k)
initial_sigma = np.ones(k)
initial_priors = np.ones(k, dtype=np.float)
initial_priors /= initial_priors.sum()

em = GaussianMixture(k, mu=initial_mu, sigma=initial_sigma, priors=initial_priors)
em.fit(data)
pred_priors, (pred_mu, pred_sigma) = em.predict()
print("\nPredicted priors: ", pred_priors)
print("Predicted mus: ", pred_mu)
print("Predicted sigmas: ", pred_sigma)

# plt.scatter(data, np.zeros_like(data), s=10, c=['blue'])
plt.scatter(data, pdfs, s=10, c=['blue'])
x = np.linspace(X_MIN-2, X_MAX+2, num_points)
for component in components:
    plt.plot(x, component.pdf(x), color='blue')
# plot gaussian mixture
plt.plot(x, [em.compute_gaussian_mixture_pdf(point) for point in x], color='red')
plt.xlim(X_MIN-2, X_MAX+2)
plt.ylim(Y_MIN, Y_MAX)
plt.show()


# if k == 2:
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#
#     mu1_coords = np.arange(X_MIN, X_MAX, 0.25)
#     mu2_coords = np.arange(X_MIN, X_MAX, 0.25)
#     mu1_grid_coords, mu2_grid_coords = np.meshgrid(mu1_coords, mu2_coords)
#     Z = np.array([em.log_likelihood_of_mu((mu1, mu2)) for mu1 in mu1_coords for mu2 in mu2_coords])
#
#     # Plot the surface.
#     surf = ax.plot_surface(mu1_grid_coords, mu2_grid_coords,
#                            Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
#     # Customize the z axis.
#     # ax.set_zlim(-1.01, 1.01)
#     # ax.zaxis.set_major_locator(LinearLocator(10))
#     # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     plt.show()