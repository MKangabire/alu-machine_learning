#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""
import math


class Poisson:
    """ class documentation"""
    def __init__(self, data=None, lambtha=1.):
        """Create a class Poisson that represents a poisson distribution"""
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

    def pmf(self, k):
      """calculates the pmf for a given number of successes"""
      k = int(k)

      if k < 0:
        return 0

      lambtha = self.lambtha
      pmf_value = (lambtha ** k) * math.exp(-lambtha) / math.factorial(k)
      return pmf_value

    
