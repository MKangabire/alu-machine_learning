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
