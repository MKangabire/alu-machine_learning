#!/usr/bin/env python3 
import numpy as np

def definiteness(matrix):
    """Check the definiteness of a matrix using its eigenvalues."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
        
    eigenvalues = np.linalg.eigvals(matrix)
    
    if np.all(eigenvalues > 0):
        return "Positive Definite"
    elif np.all(eigenvalues < 0):
        return "Negative Definite"
    elif np.all(eigenvalues >= 0):
        return "Positive Semi-Definite"
    elif np.all(eigenvalues <= 0):
        return "Negative Semi-Definite"
    else:
        return "Indefinite"
