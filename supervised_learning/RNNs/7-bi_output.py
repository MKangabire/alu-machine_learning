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

        # Output parameters
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the forward hidden state for one time step.

        Args:
            h_prev (np.ndarray): shape (m, h), previous hidden state
            x_t (np.ndarray): shape (m, i), input at time t

        Returns:
            np.ndarray: shape (m, h), next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """
        Calculates the backward hidden state for one time step.

        Args:
            h_next (np.ndarray): shape (m, h), next hidden state
            x_t (np.ndarray): shape (m, i), input at time t

        Returns:
            np.ndarray: shape (m, h), previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whb) + self.bhb)

    def output(self, H):
        """
        Calculates all outputs for the bidirectional RNN.

        Args:
            H (np.ndarray): shape (t, m, 2 * h), concatenated hidden states

        Returns:
            Y (np.ndarray): shape (t, m, o), outputs
        """
        t, m, _ = H.shape
        Y = []

        for time_step in range(t):
            h_concat = H[time_step]  # shape (m, 2*h)
            y_t = np.matmul(h_concat, self.Wy) + self.by  # shape (m, o)
            # Apply softmax
            y_t_exp = np.exp(y_t - np.max(y_t, axis=1, keepdims=True))  # stability
            y_t_softmax = y_t_exp / np.sum(y_t_exp, axis=1, keepdims=True)
            Y.append(y_t_softmax)

        return np.array(Y)  # shape (t, m, o)
