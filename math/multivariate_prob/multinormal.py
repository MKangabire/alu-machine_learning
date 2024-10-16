#!/usr/bin/env python3
"""A class that calculates the mean, covariance, and PDF of a data set."""
import numpy as np

class MultiNormal:
    """A class that represents a multivariate normal distribution."""
    
    def __init__(self, data):
        """Initializes the multivariate normal distribution."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        n, d = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.data = data
        self.mean = np.mean(data, axis=0, keepdims=True).reshape(d, 1)
        centered_data = data - self.mean
        self.cov = np.dot(centered_data.T, centered_data) / (n - 1)

    def pdf(self, x):
        """Calculates the probability density function of the multivariate normal distribution."""
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise TypeError("x must be a 1D numpy.ndarray")
        if x.shape[0] != self.data.shape[1]:
            raise ValueError("x must have the same number of elements as the number of features in data")

        d = self.data.shape[1]
        diff = x.reshape(d, 1) - self.mean

        det_cov = np.linalg.det(self.cov)
        denominator = np.sqrt((2 * np.pi) ** d * det_cov)

        exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff)

        return (1 / denominator) * np.exp(exponent)[0][0]
