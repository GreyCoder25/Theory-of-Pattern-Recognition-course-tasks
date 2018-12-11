import matplotlib.pyplot as plt
from utils import DatasetGenerator
from perceptron import PerceptronClassifier
import numpy as np

discr_func = 'ellipse'
data_generator = DatasetGenerator()
X, y = data_generator.generate_separable_data(num_points=2000, discr_func=discr_func)

# clf = PerceptronClassifier(discr_func=discr_func)
# clf.fit(X, y)

plt.xlim(-14, 14)
plt.ylim(-14, 14)
# plot data points
plt.scatter(X[:, 0], X[:, 1], s=25, c=y)
# plot separating curve
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xlist = np.linspace(xmin, xmax, 100)
ylist = np.linspace(ymin, ymax, 100)
XX, YY = np.meshgrid(xlist, ylist)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.score(xy).reshape(XX.shape)
# plt.contour(XX, YY, Z, levels=[0], colors=['r'])
plt.show()
