#!/usr/bin/env python3
"""recurrent neural networks"""


import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell for one time step.
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the output
        """
        # Weight matrices
        self.Wz = np.random.randn(i + h, h)  # Update gate weights
        self.Wr = np.random.randn(i + h, h)  # Reset gate weights
        self.Wh = np.random.randn(i + h, h)  # Candidate hidden state weights
        self.Wy = np.random.randn(h, o)      # Output weights

        # Bias vectors
        self.bz = np.zeros((1, h))  # Update gate bias
        self.br = np.zeros((1, h))  # Reset gate bias
        self.bh = np.zeros((1, h))  # Candidate hidden state bias
        self.by = np.zeros((1, o))  # Output bias

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (ndarray): Previous hidden state of shape (m, h)
            x_t (ndarray): Current input data of shape (m, i)

        Returns:
            h_next (ndarray): Next hidden state
            y (ndarray): Output of the cell
        """
        # Concatenate previous hidden state and current input
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.matmul(concatenated, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.matmul(concatenated, self.Wr) + self.br)

        # Apply reset gate to hidden state
        concatenated_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concatenated_reset, self.Wh) + self.bh)

        # Final hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_hat

        # Output using softmax
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
