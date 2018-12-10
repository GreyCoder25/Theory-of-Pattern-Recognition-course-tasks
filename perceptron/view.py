import matplotlib.pyplot as plt
from utils import DatasetGenerator
from perceptron import PerceptronClassifier
import numpy as np

data_generator = DatasetGenerator()
X, y = data_generator.generate_separable_data(num_points=1000, discr_func='linear')

clf = PerceptronClassifier()
clf.fit(X, y)

# plot data points
plt.scatter(X[:, 0], X[:, 1], s=25, c=y)
# plot separating curve
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xlist = np.linspace(xmin, xmax, 30)
ylist = np.linspace(ymin, ymax, 30)
XX, YY = np.meshgrid(xlist, ylist)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.score(xy).reshape(XX.shape)
plt.contour(XX, YY, Z, levels=[0], colors=['r'])
plt.show()