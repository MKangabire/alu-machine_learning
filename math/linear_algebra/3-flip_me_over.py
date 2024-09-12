#!/usr/bin/env python3
def matrix_transpose(matrix):
    """returns the transpose of the matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]