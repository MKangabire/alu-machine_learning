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
