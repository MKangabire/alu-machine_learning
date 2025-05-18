#!/usr/bin/env python3
"""recurrent neural networks"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances (length el), 
        X (ndarray): Input data of shape (t, m, i)
            - t: Number of time steps
            - m: Batch size
            - i: Input feature size
        h_0 (ndarray): Initial hidden states of shape (el, m, h)
            - el: Number of layers
            - h: Hidden state size

    Returns:
        H (ndarray): All hidden states of shape (t + 1, el, m, h)
        Y (ndarray): All outputs of shape (t, m, o)
    """
    t, m, _ = X.shape
    el = len(rnn_cells)
    _, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]  # Output dimension from the last RNNCell

    # Initialize arrays to store all hidden states and outputs
    H = np.zeros((t + 1, el, m, h))
    Y = np.zeros((t, m, o))

    # Set the initial hidden state
    H[0] = h_0

    # Loop through each time step
    for time_step in range(t):
        x = X[time_step]

        for layer in range(el):
            # Pass x and previous hidden state through the current RNNCell
            h_prev = H[time_step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)

            # Store new hidden state
            H[time_step + 1, layer] = h_next

            # Output of current layer is input to next layer
            x = h_next

        # Store final layer's output
        Y[time_step] = y

    return H, Y
