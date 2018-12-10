import numpy as np
import numpy.random as rnd


class DatasetGenerator:

    def __init__(self, epsilon=10):
        self._epsilon = epsilon

    def generate_separable_data(self, num_points=100, dim=2, discr_func='linear', distr='uniform',
                                data_intervals=np.array([(-50, 50), (-50, 50)]), coeffs=None):
        if dim != data_intervals.shape[0]:
            raise ValueError('Number of data intervals is not equal to number of data dimensions')

        if discr_func == 'linear':
            coeffs = -0.5 + rnd.rand(dim + 1)
            return self._generate_linearly_separable_data(num_points, dim, distr, data_intervals, coeffs)

    def _generate_linearly_separable_data(self, num_points, dim, distr, data_intervals, coeffs):
        if dim + 1 != coeffs.size:
            raise ValueError("Size of coeffs array doesn't match the number of dimensions (it have to be equal to "
                             "dim + 1)")

        X = np.ones((num_points, dim + 1))
        if distr == 'uniform':
            for index, interval in enumerate(data_intervals):
                X[:, index + 1] = rnd.uniform(interval[0], interval[1], X.shape[0])
        y = (X.dot(coeffs) > 0).astype(int)

        # next code is dirty hack for make data convenient for separating
        shift = np.zeros(dim + 1)
        slope = -coeffs[1] / coeffs[2]
        if (slope > 0 and coeffs[1]) > 0 or (slope < 0 and coeffs[1] < 0):
            shift[dim-1] = -slope
            shift[dim] = self._epsilon
        else:
            shift[dim-1] = slope
            shift[dim] = -self._epsilon
        X[y == 0] += shift

        return X[:, 1:], y

