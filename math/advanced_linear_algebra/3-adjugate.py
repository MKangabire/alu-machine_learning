#!/usr/bin/env python3
"""Calculate the cofactor and adjugate (adjoint) matrix of a matrix."""

def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix."""
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a list of lists")

    rows = len(matrix)
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix) or rows != cols:
        raise ValueError("matrix must be a non-empty square matrix")
    
    def get_submatrix(m, row, col):
        """Get the submatrix excluding the specified row and column."""
        return [r[:col] + r[col+1:] for i, r in enumerate(m) if i != row]
    
    def determinant(m):
        """Calculate the determinant of a matrix."""
        if len(m) == 1:
            return m[0][0]
        
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        
        det = 0
        for col in range(len(m)):
            submatrix = get_submatrix(m, 0, col)
            det += ((-1) ** col) * m[0][col] * determinant(submatrix)
        return det
    
    cofactor_matrix = [[(((-1) ** (i + j)) * determinant(get_submatrix(matrix, i, j)))
                        for j in range(cols)] for i in range(rows)]
    return cofactor_matrix

def transpose(matrix):
    """Transpose the matrix."""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def adjugate(matrix):
    """Calculate the adjugate (adjoint) of a matrix."""
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = transpose(cofactor_matrix)
    return adjugate_matrix
