#!/usr/bin/env python3
"""Perform matrix multiplication."""


import numpy as np


def np_matmul(mat1, mat2):
    """Perform matrix multiplication."""
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    solution = np.matmul(mat1, mat2)
    return solution
