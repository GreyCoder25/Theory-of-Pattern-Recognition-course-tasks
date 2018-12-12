import numpy as np
import numpy.linalg as la


class PerceptronClassifier:

    def __init__(self, discr_func='linear'):
        self._discr_func = discr_func

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._m, self._n = self._X.shape
        self._batch_size = self._m
        self._initialize_weight_vector()
        self._augment_dataset()
        self._learn_weights()

    def score(self, x):
        x = self._augment_feature_vector(x)
        return x.dot(self._w)

    def _learn_weights(self):
        pattern_index = 0
        correct_counter = 0
        is_ellipse = False
        while correct_counter != self._m:
            if self._X[pattern_index].dot(self._w) <= 0 and self._y[pattern_index] == 1:
                self._w += self._X[pattern_index]
                correct_counter = 0
                is_ellipse = False
            elif self._X[pattern_index].dot(self._w) >= 0 and self._y[pattern_index] == 0:
                self._w -= self._X[pattern_index]
                correct_counter = 0
                is_ellipse = False
            else:
                correct_counter += 1

            if self._discr_func == 'ellipse' and correct_counter == self._m:
                corrections_performed = self._check_and_correct_eigenvalues()
                if corrections_performed:
                    correct_counter = 0
                    is_ellipse = False
                else:
                    is_ellipse = True

            if correct_counter == self._m and is_ellipse:
                return
            pattern_index = (pattern_index + 1) % self._m

    def _check_and_correct_eigenvalues(self):
        p = self._w[self._n + 1:]
        p_mat = np.empty((self._n, self._n))
        p_index = 0
        for i in range(self._n):
            for j in range(i, self._n):
                if i == j:
                    p_mat[i, j] = p[p_index]
                elif i != j:
                    p_mat[i, j] = p[p_index] / 2
                    p_mat[j, i] = p[p_index] / 2
                p_index += 1
        w, v = la.eig(p_mat)
        for ev_index, eigen_val in enumerate(w):
            if eigen_val < 0:
                eig_vec = v[:, ev_index]
                ev_pattern = np.zeros(self._w.shape)
                ev_pattern[self._n + 1:] = np.array([eig_vec[i] * eig_vec[j] for i in range(self._n)
                                                     for j in range(i, self._n)])
                self._w += ev_pattern
                return True
        return False

    def _augment_feature_vector(self, x):
        if self._discr_func == 'linear':
            x_augmented = np.ones((x.shape[0], self._n + 1))
            x_augmented[:, 1:] = x
        elif self._discr_func in ['quadratic', 'ellipse']:
            x_augmented = np.ones((x.shape[0], int((self._n + 1)*(self._n + 2)/2)))
            x_augmented[:, 1:self._n + 1] = x
            x_augmented[:, self._n + 1:] = np.array([x[:, i]*x[:, j] for i in range(self._n) for j in range(i, self._n)]).T
        return x_augmented

    def _augment_dataset(self):
        self._X = self._augment_feature_vector(self._X)

    def _initialize_weight_vector(self):
        weight_vector_size = 0
        if self._discr_func == 'linear':
            weight_vector_size = self._n + 1
        elif self._discr_func == 'sphere':
            weight_vector_size = self._n + 2
        elif self._discr_func in ['ellipse', 'quadratic']:
            weight_vector_size = int((self._n**2 + 3*self._n + 2) / 2)
        self._w = np.zeros(weight_vector_size)
