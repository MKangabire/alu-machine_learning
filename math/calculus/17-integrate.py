#!/usr/bin/env python3
"""Write a function def poly_integral(poly, C=0): that calculates"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

i    Args:
        poly (list): A list of coefficients representing a polynomial.
        C (int, optional): The integration constant. Defaults to 0.

    Returns:
    list: A new list of coefficients representing the
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    if not isinstance(C, int):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        if coeff == 0:
            integral.append(0)
        else:
            integral_coeff = coeff / (i + 1)

            if integral_coeff.is_integer():
                integral.append(int(integral_coeff))
            else:
                integral.append(integral_coeff)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
