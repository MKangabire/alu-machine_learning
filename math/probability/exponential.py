#!/usr/bin/env python3
"""Create a class Exponential that represents an exponential distribution"""


class Exponential:
    """Class that represents an exponential distribution"""
    
    def __init__(self, data=None, lambtha=1.):
        """Initialize the Exponential class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculate the PDF for a given value of x"""
        if x < 0:
            return 0
        e = 2.7182818285  # Euler's number approximation
        pdf_value = self.lambtha * e ** (-self.lambtha * x)
        return pdf_value
