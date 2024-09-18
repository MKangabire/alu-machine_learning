#!/usr/bin/env python3
"""A function that calculates the determinant of a matrix without importing any libraries."""


def determinant(matrix):
    """A function that calculates the determinant of a matrix."""
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for i in range(len(matrix)):
        submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        
        cofactor = (-1) ** i * matrix[0][i]
        det += cofactor * determinant(submatrix)
    
    return det