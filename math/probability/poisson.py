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

    def pmf(self, k):
        """Calculates the PMF for a given number of successes (k)"""
        k = int(k)

        if k < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        pmf_value = (lambtha ** k) * e ** (-lambtha) / self.factorial(k)
        return pmf_value

    def cdf(self, k):
        """Calculates the CDF for a given number of successes (k)"""
        k = int(k)

        if k < 0:
            return 0
        cdf_value = 0  # Initialize cdf_value
        e = 2.7182818285
        lambtha = self.lambtha
        for i in range(0, k + 1):
            cdf_value += (lambtha ** i) * e ** (-lambtha) / self.factorial(i)
        return cdf_value
