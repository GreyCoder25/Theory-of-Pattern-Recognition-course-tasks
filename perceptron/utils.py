import numpy as np
import numpy.random as rnd


class DatasetGenerator:

    def __init__(self, epsilon=0.01):
        self._epsilon = epsilon

    def generate_separable_data(self, num_points=100, dim=2, discr_func='linear', distr='uniform',
                                data_intervals=np.array([(-10, 10), (-10, 10)]), coeffs=None):
        if dim != data_intervals.shape[0]:
            raise ValueError('Number of data intervals is not equal to number of data dimensions')

        self._discr_func = discr_func
        if discr_func == 'linear':
            coeffs = -1 + 2*rnd.rand(dim + 1)
            return self._generate_linearly_separable_data(num_points, dim, distr, data_intervals, coeffs)
        elif discr_func in ['quadratic', 'ellipse']:
            return self._generate_quadratically_separable_data(num_points, dim, distr, data_intervals, coeffs)

    def _generate_linearly_separable_data(self, num_points, dim, distr, data_intervals, coeffs):
        if (coeffs is not None) and dim + 1 != coeffs.size:
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

        return X[:, 1:dim+1], y

    def _generate_quadratically_separable_data(self, num_points, dim, distr, data_intervals, coeffs):
        augmented_dim = int((dim + 1)*(dim + 2)/2)
        if (coeffs is not None) and coeffs.size != augmented_dim:
            raise ValueError("Size of coeffs array doesn't match the number of dimensions (it have to be equal to "
                             "(dim + 1)*(dim + 2)/2)")
        coeffs = -0.5 + rnd.rand(augmented_dim)
        if self._discr_func == 'ellipse':
            rand_mat = -0.5 + rnd.rand(dim, dim)
            rand_pos_mat = rand_mat.dot(rand_mat.T)
            coeffs[dim+1:] = np.array([rand_pos_mat[i, j] if i == j else rand_pos_mat[i, j] + rand_pos_mat[j, i]
                                       for i in range(dim) for j in range(i, dim)])

        X = np.ones((num_points, augmented_dim))
        point_index = 0
        while point_index != num_points:
            X[point_index][1:dim+1] = rnd.uniform(*data_intervals.T)
            X[point_index][dim+1:] = np.array([X[point_index][i]*X[point_index][j] for i in range(1, dim+1)
                                               for j in range(i, dim+1)])

            if np.abs(X[point_index].dot(coeffs)) > self._epsilon:
                point_index += 1
            # if X[point_index].dot(coeffs) > self._epsilon or X[point_index].dot(coeffs) < 0:
            #     point_index += 1

        y = (X.dot(coeffs) > 0).astype(int)
        return X[:, 1:dim + 1], y




