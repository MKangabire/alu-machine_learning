#!/usr/bin/env python3
"""a function that calculates the likelihood of obtaining the data"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of

    Args:
        x (int): Number of patients with severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array of hypothetical probabilities.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(k):
        if k == 0 or k == 1:
            return 1
        result = 1
        for i in range(2, k + 1):
            result *= i
        return result

    binom_coeff = factorial(n) / (factorial(x) * factorial(n - x))

    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))
    return likelihoods
