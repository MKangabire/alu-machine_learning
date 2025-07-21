#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on the dataset X to reduce its dimensionality
    while maintaining the desired fraction of variance.

    Parameters:
    - X: numpy.ndarray of shape (n, d)
        n: number of data points
        d: original number of features (dimensions)
    - var: float, the fraction of the total variance to preserve (default is 0.95)

    Returns:
    - W: numpy.ndarray of shape (d, nd)
        The projection matrix to reduce X's dimensionality to nd
    """
    # Step 1: Compute the covariance matrix of X
    cov_matrix = np.cov(X, rowvar=False)

    # Step 2: Perform eigen-decomposition on the covariance matrix
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Step 3: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    # Step 4: Compute the total variance and cumulative variance ratio
    total_var = np.sum(eig_vals)
    cum_var_ratio = np.cumsum(eig_vals) / total_var

    # Step 5: Find the minimum number of dimensions to retain the required variance
    nd = np.searchsorted(cum_var_ratio, var) + 1

    # Step 6: Select the top nd eigenvectors (principal components)
    W = eig_vecs[:, :nd]

    return W
