import numpy as np


class PerceptronClassifier:

    def __init__(self, discr_func='linear'):
        self._weights = []
        self._X = []
        self._y = []

    def fit(self, X, y):

    def _augment_feature_vector(self, x):