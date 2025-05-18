#!/usr/bin/env python3
"""recurrent neural networks"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN over multiple time steps.

    Args:
        rnn_cell (RNNCell): An instance of the RNNCell class used 
        X (ndarray): Input data for the RNN, of shape (t, m, i)
                     - t: number of time steps
                     - m: batch size
                     - i: input feature size
        h_0 (ndarray): Initial hidden state, of shape (m, h)
                      - h: hidden state size

    Returns:
        tuple:
            H (ndarray): Hidden states for all time steps, shape (t + 1, m, h)
            Y (ndarray): Outputs for all time steps, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    # Initialize arrays to hold hidden states and outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Loop through each time step
    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
