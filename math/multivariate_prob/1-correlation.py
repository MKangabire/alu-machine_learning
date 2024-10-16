import numpy as np
#!/usr/bin/env python3
"""A function that calculates the Correlation of a data set."""


def correlation(C):
    """Calculates the correlation matrix from a covariance matrix.
    
    Args:
    C (numpy.ndarray): Covariance matrix of shape (d, d)
    
    Returns:
    numpy.ndarray: Correlation matrix of shape (d, d)
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))

    if np.any(std_devs == 0):
        raise ValueError("Standard deviation cannot be zero")
    
    std_outer = np.outer(std_devs, std_devs)

    corr_matrix = C / std_outer

    return corr_matrix
