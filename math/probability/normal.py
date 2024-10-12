#!/usr/bin/env python3
"""Create a class Normal that represents a normal distribution"""


class Normal:
    """Class that represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the Normal class"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate the z-score for a given value of x"""
        z_score_value = (x - self.mean) / self.stddev
        return z_score_value

    def x_value(self, z):
        """Calculate the x-value for a given z-score"""
        x_value = z * self.stddev + self.mean
        return x_value

    def pdf(self, x):
        """Calculate the PDF for a given value of x"""
        pi = 3.1415926535897
        e = 2.7182818285
        pdf_value = (1 / (self.stddev * (2 * pi) ** 0.5)) * e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return pdf_value

    def erf(self, z):
        """Approximate the error function (erf) using a Maclaurin series"""
        pi = 3.1415926535897
        sum = 0
        term = z  # first term in the series
        for n in range(20):  # higher range gives more accurate result
            sum += term
            term *= -z ** 2 / (n + 1) / (2 * n + 3)
        erf = (2 / (pi ** 0.5)) * sum
        return erf

    def cdf(self, x):
        """Calculate the CDF for a given value of x"""
        pi = 3.1415926535897
        e = 2.7182818285
        cdf_value = 0.5 * (1 + erf((x - self.mean) / (self.stddev * (2 ** 0.5))))
        return cdf_value
