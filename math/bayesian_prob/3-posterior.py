#!/usr/bin/env python3
"""A function that calculates the posterior probability"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    A function that calculates the posterior probability
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_n_x = np.math.factorial(n - x)
    binomial_coeff = fact_n / (fact_x * fact_n_x)

    likelihoods = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihoods * Pr
    marginal_prob = np.sum(intersection)
    posterior = intersection / marginal_prob

    return posterior
