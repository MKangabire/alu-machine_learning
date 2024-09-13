#!/usr/bin/env python3
"""Perform that concatenates two matrices along a specific axes."""


import numpy as np

def np_cat(mat1, mat2, axis=0):
    """Perform that concatenates two matrices along a specific axes."""
    np.concatenate((mat1, mat2), axis=axis)
