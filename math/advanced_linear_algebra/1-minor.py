import numpy as np

def minor(matrix):
    """Calculate the minor matrix of a matrix."""
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    matrix = np.array(matrix)
    
    if matrix.size == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("matrix must be a non-empty square matrix")

    def get_minor(m, row, col):
        """Get the minor of element at (row, col)."""
        submatrix = np.delete(np.delete(m, row, axis=0), col, axis=1)
        return np.linalg.det(submatrix)
    
    minor_matrix = np.array([[get_minor(matrix, i, j) for j in range(cols)] for i in range(rows)])
    return minor_matrix.tolist() 
