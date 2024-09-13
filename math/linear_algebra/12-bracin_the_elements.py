#!/usr/bin/env python3
"""Function that performs element-wise."""


import numpy as np

def np_elementwise(mat1, mat2):
    """Perform that performs element-wise."""
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    sum_result = mat1 + mat2
    diff_result = mat1 - mat2
    prod_result = mat1 * mat2
    quot_result = mat1 / mat2

    return (sum_result, diff_result, prod_result, quot_result)
