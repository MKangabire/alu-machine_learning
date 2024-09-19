#!/usr/bin/env python3
"""A function that calculates the determinant of a matrix without importing any libraries."""


def determinant(matrix):
    """Calculate the determinant of a matrix."""
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    if matrix == [[]]:
        return 1 

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices (Laplace expansion)
    det = 0
    for col in range(len(matrix)):
        submatrix = [[matrix[i][j] for j in range(len(matrix)) if j != col] for i in range(1, len(matrix))]
        det += ((-1) ** col) * matrix[0][col] * determinant(submatrix)

    return det
