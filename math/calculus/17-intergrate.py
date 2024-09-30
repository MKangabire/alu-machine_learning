#!/usr/bin/env python3
"""Write a function def poly_integral(poly, C=0): that calculates"""

def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.
        C (int, optional): The integration constant. Defaults to 0.

    Returns:
        list: A new list of coefficients representing the integral of the polynomi
    """
    if not isinstance(poly, list):
        return None

    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    if not isinstance(C, int):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        integral.append(coeff / (i + 1))

    while integral and integral[-1] == 0:
        integral.pop()

    return integral
