import numpy as np
from sklearn.neighbors import KDTree
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
print(X)
tree = KDTree(X, leaf_size=2)
print(tree)# doctest: +SKIP
print(X[:1])
dist, ind = tree.query(X[:2], k=3)                # doctest: +SKIP
print(ind)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors
