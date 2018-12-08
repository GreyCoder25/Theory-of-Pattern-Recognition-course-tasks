import numpy as np
import numpy.random as rnd


class DatasetGenerator:

    def __init__(self):
        self._epsilon = 10

    def generate_separable_data(self, num_points=100, dim=2, discr_func='linear', distr='uniform',
                                data_intervals=np.array([(-50, 50), (-50, 50)]), coeffs=np.array([3, 8, -1])):
        if dim != data_intervals.shape[0]:
            raise ValueError('Number of data intervals is not equal to number of data dimensions')

        if discr_func == 'linear':
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

        shift = np.zeros(dim + 1)
        shift[dim-1] = coeffs[1] / coeffs[2]        # dirty hack
        shift[dim] = self._epsilon
        X[y == 0] += shift

        return X, y

