#!/usr/bin/env python3
"""a function that sums"""


def summation_i_squared(n):
    """a function that calculates the sum of 1 to n"""
    if not isinstance(n, int):
        return None
    solution = 0
    for i in range(1,n+1):
        b = i**2
        solution += b
    return solution
