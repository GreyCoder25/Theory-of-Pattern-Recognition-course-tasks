import numpy as np


class PerceptronClassifier:

    def __init__(self, discr_func='linear'):
        self._discr_func = discr_func

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._m, self._n = self._X.shape
        self._initialize_weight_vector()
        self._augment_dataset()
        self._learn_weights()

    def score(self, x):
        x = self._augment_feature_vector(x)
        return x.dot(self._w)

    def _learn_weights(self):
        pattern_index = 0
        correct_counter = 0
        while correct_counter != self._m:
            if self._X[pattern_index].dot(self._w) <= 0 and self._y[pattern_index] == 1:
                self._w += self._X[pattern_index]
                correct_counter = 0
            elif self._X[pattern_index].dot(self._w) >= 0 and self._y[pattern_index] == 0:
                self._w -= self._X[pattern_index]
                correct_counter = 0
            else:
                correct_counter += 1
            if correct_counter == self._m:
                return
            pattern_index = (pattern_index + 1) % self._m

    def _augment_feature_vector(self, x):
        if self._discr_func == 'linear':
            x_augmented = np.ones((x.shape[0], self._n + 1))
            x_augmented[:, 1:] = x
        return x_augmented

    def _augment_dataset(self):
        if self._discr_func == 'linear':
            data = self._X
            self._X = np.ones((self._m, self._n + 1))
            self._X[:, 1:] = data

    def _initialize_weight_vector(self):
        weight_vector_size = 0
        if self._discr_func == 'linear':
            weight_vector_size = self._n + 1
        elif self._discr_func == 'sphere':
            weight_vector_size = self._n + 2
        elif self._discr_func in ['ellipse', 'quadratic']:
            weight_vector_size = int((self._n**2 + 3*self._n + 2) / 2)
        self._w = np.zeros(weight_vector_size)
