#!/usr/bin/env python3
"""a function that calculates the likelihood of obtaining the data"""
import numpy as np


def likelihood(x, n, P):
  """a function that calculates the likelihood of obtaining the data"""
  if not isinstance(n, int) or x < 0:
    raise ValueError("n must be a positive integer")
  if x > n:
    raise ValueError("x cannot be greater than n")
  if np.any((P < 0) | (P > 1)):
    raise ValueError("All values in P must be in the range [0, 1]")
  if not isinstance(P, np.ndarray) or P.ndim != 1:
    raise TypeError("P must be a 1D numpy.ndarray")
  def factorial(k):
    """finds the factorial"""
    if k == 0 or k == 1:
        return 1
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result
  binomial_coef = factorial(n) / (factorial(x) * factorial(n - x))
  likelihood_value = binomial_coef * (P ** x) * ((1 - P) ** (n - x))
  return likelihood_value