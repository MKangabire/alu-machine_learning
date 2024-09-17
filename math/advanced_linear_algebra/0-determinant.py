#!/usr/bin/env python3
"""A functiom that calculates the determinant of a matrix"""


import numpy as np
def determinant(matrix):
    """A functiom that calculates the determinant of a matrix."""
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if not matrix or not any(row for row in matrix):
        return 1

    np_matrix = np.array(matrix)
    sol = np.linalg.det(np_matrix)
    return sol