#!/usr/bin/env python3
"""Regularization"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2"""
    l2_norm = 0
    for i in range(1, L + 1):
        l2_norm += np.sum(np.square(weights[f'W{i}']))

    l2_cost = cost + (lambtha / (2 * m)) * l2_norm
    return l2_cost
   