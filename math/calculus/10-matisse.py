#!/usr/bin/env python3
"""that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.
        C (int, optional): The integration constant. Defaults to 0.

    Returns:
        list: A new list of coefficients representing the integral 
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None
    integral = []
    for i, coeff in enumerate(poly):
        integral.append(coeff / (i + 1))
