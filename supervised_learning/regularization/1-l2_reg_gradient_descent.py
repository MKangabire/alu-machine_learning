#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """a function that updates the weights and biases of a neural network
    using gradient descent wit L2 regularization"""
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f'A{i-1}'] if i > 1 else cache['A0']
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        dw = (1 / m) * np.dot(dZ. A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        weights[f'W{i}'] -= alpha * dw
        weights[f'b{i}'] -= alpha * db

        if i > 1:
            dZ = np.dot(W.T, dZ) * (1 - np.square(cache[f'A{i-1}']))
        