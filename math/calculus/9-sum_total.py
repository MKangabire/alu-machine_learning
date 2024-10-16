#!/usr/bin/env python3
"""a function that sums"""


def summation_i_squared(n):
    """a function that calculates the sum of 1 to n"""
    if isinstance(n, int) and n > 0:
        return n*(n+1)*(2*n+1)//6
    else:
        return None
