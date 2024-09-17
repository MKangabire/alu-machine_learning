#!/usr/bin/env python3
"""Concatenate two matrices along a specific axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""
    new_matrix = [row[:] for row in mat1]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return new_matrix + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(new_matrix)):
            new_matrix[i] += mat2[i]
        return new_matrix
    return None
