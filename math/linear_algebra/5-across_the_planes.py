#!/usr/bin/env python3
"""Adds 2 matrices element."""


def add_matrices2D(mat1, mat2):
    """Add 2 matrices element."""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    new_matrix = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    return new_matrix
