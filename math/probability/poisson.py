#!/usr/bin/env python3
"""Create a class Poisson that represents a Poisson distribution"""


class Poisson:
    """Class that represents a Poisson distribution"""    
    def __init__(self, data=None, lambtha=1.):
        """Initialize the Poisson class"""
        if data is None:
            # Use given lambtha if data is not provided
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate the data input
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha from data (mean of the data)
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """Calculate factorial of n (n!)"""
        if n < 0:
            return 0
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp(self, x):
        """Calculate e^x using a Taylor series approximation"""
        result = 1.0  # Start with the first term of the series
        term = 1.0  # Initialize the term to 1 (x^0 / 0!)
        for n in range(1, 201):  # 100 terms for a good approximation
            term *= x / n  # Calculate the next term
            result += term  # Add the term to the result
        return result

    def pmf(self, k):
        """Calculates the PMF for a given number of successes (k)"""
        k = int(k)

        if k < 0:
            return 0
        lambtha = self.lambtha
        pmf_value = (lambtha ** k) * self.exp(-lambtha) / self.factorial(k)
        return pmf_value

    def cdf(self, k):
      """calculates the cdf"""
      k = int(k)

      if k < 0:
        return 0
      for i in range(0, k + 1):
        lambtha = self.lambtha
        cdf_value += (lambtha ** i) * sel.exp ** (-lambtha) / self.factor(i)
        return cdf_value
      
