import matplotlib.pyplot as plt
from utils import DatasetGenerator

data_generator = DatasetGenerator()
X, y = data_generator.generate_separable_data(num_points=500, discr_func='linear')
plt.scatter(X[:, 1], X[:, 2], s=25, c=y)
plt.show()