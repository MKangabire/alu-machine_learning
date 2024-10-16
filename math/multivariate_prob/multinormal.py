#!/usr/bin/env python3
"""A function that calculates the mean and covariance of a data set."""
import numpy as np


class MultiNormal:
    """a class that represents a multinormal distribution."""
    def __init__(self, data):
        """initialises the multinormal distribution."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.data = data
        self.mean = np.mean(data, axis=1, keepdims=True).reshape(d, 1)
        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (n - 1)
    #!/usr/bin/env python3
"""A function that calculates the mean and covariance of a data set."""
import numpy as np


class Multinormal:
  """a class that represents a multinormal distribution."""
  def __init__(self, data):
    """initialises the multinormal distribution."""
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy.ndarray")
    n,d = data.shape
    if n < 2:
        raise ValueError("data must contain multiple data points")
    self.data = data
    self.mean = np.mean(data, axis=0).reshape(d, 1)
    centered_data = data - np.mean(data, axis=0)
    self.cov = np.dot(centered_data.T, centered_data) / (n - 1)

  def pdf(self, x):
    """calculates the probability density function of the multinormal distribution."""
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise TypeError("x must be a 2D numpy.ndarray")
    X_shape = x.shape
    if len(X_shape) != 2 or X_shape[1] != self.data.shape[1]:
        raise ValueError("x must have the same number of columns as data")

    d = self.data.shape[1]
    diff = x.reshape(d, 1) - self.mean
    det_cov = np.linalg.det(self.cov)
    denominator = np.sqrt((2 * np.pi) ** d * det_cov)
    exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff)
    pdf = (1 / denominator) * np.exp(exponent)
    return pdf