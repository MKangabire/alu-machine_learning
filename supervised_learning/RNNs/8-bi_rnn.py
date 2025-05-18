#!/usr/bin/env python3
"""Bidirectional RNN forward propagation"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Parameters:
    - bi_cell: instance of BidirectionalCell
    - X: np.ndarray of shape (t, m, i) with input data
    - h_0: np.ndarray of shape (m, h)
    - h_t: np.ndarray of shape (m, h)

    Returns:
    - H: np.ndarray of shape (t, m, 2*h),
    - Y: np.ndarray of shape (t, m, o), outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize containers for hidden states
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    # Forward pass
    h_prev = h_0
    for time in range(t):
        h_prev = bi_cell.forward(h_prev, X[time])
        Hf[time] = h_prev

    # Backward pass
    h_next = h_t
    for time in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[time])
        Hb[time] = h_next

    # Concatenate hidden states
    H = np.concatenate((Hf, Hb), axis=2)

    # Compute output
    Y = bi_cell.output(H)

    return H, Y
