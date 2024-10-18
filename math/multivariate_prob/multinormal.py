#!/usr/bin/env python3
"""A class that calculates the mean, covariance, and PDF of a data set."""
import numpy as np


class MultiNormal:
    """A class that represents a multivariate normal distribution."""
    
    def __init__(self, data):
        """Initializes the multivariate normal distribution."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape  # Number of dimensions (d) and number of data points (n)
        if n < 2:
            raise ValueError("data must contain multiple data points")
        
        self.data = data
        self.mean = np.mean(data, axis=1, keepdims=True)  # Compute mean along axis 1 for each dimension
        centered_data = data - self.mean  # Subtract mean from data
        self.cov = np.dot(centered_data, centered_data.T) / (n - 1)  # Covariance matrix calculation

    def pdf(self, x):
        """Calculates the probability density function of the"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        
        d = self.data.shape[0]  # Number of dimensions
        if x.shape != (d, 1):  # Check if x has the shape (d, 1)
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean

        det_cov = np.linalg.det(self.cov)
        denominator = np.sqrt((2 * np.pi) ** d * det_cov)

        exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff)

        return (1 / denominator) * np.exp(exponent)[0][0]
