#!/usr/bin/env python3
"""This module defines a function to compute the shape of a matrix."""


def matrix_shape(matrix):
    """Return the shape of the matrix."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
