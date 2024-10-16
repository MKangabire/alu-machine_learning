#!/usr/bin python3
import numpy as np
from multinormal import MultiNormal

data = np.random.multivariate_normal([5, -4, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 10000).T
mn = MultiNormal(data)
print("Mean:\n", mn.mean)
print("Covariance matrix:\n", mn.cov)