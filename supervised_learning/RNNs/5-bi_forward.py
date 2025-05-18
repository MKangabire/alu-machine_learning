#!/usr/bin/env python3
"""recurrent neural networks"""


import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell.
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden states
            o (int): Dimensionality of the outputs
        """
        # Forward direction parameters
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Backward direction parameters
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Output parameters (after concatenating both directions)
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step in

        Args:
            h_prev (ndarray): Previous hidden state of shape (m, h)
            x_t (ndarray): Input data at time t of shape (m, i)

        Returns:
            h_next (ndarray): Next hidden state of shape (m, h)
        """
        # Concatenate input and previous hidden state
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.matmul(concatenated, self.Whf) + self.bhf)
        return h_next
