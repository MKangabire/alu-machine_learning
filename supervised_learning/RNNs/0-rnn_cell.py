#!/usr/bin/env python3
"""recurrent neural networks"""


import numpy as np


class RNNCell:
    """
    Represents a simple RNN (Recurrent Neural Network) cell.
    Performs forward propagation for a single time step.

    Attributes:
        bh (ndarray): Bias for the hidden state (shape: (1, h)).
        Wy (ndarray): Weights for the output (shape: (h, o)).
        by (ndarray): Bias for the output (shape: (1, o)).
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """

        self.Wh = np.random.randn(i + h, h)   # Shape: (i + h, h)
        self.bh = np.zeros((1, h))            # Shape: (1, h)

        # Weights and bias for output computation
        self.Wy = np.random.randn(h, o)       # Shape: (h, o)
        self.by = np.zeros((1, o))            # Shape: (1, o)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (ndarray): Previous hidden state (shape: (m, h)).
            x_t (ndarray): Current input data (shape: (m, i)).

        Returns:
            tuple:
                h_next (ndarray): Next hidden state (shape: (m, h)).
                y (ndarray): Output of the cell (shape: (m, o)).
        """
        # Concatenate previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)  # Shape: (m, i+h)

        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)  # Shape: (m, h)

        # Compute raw output
        y_linear = np.dot(h_next, self.Wy) + self.by  # Shape: (m, o)

        # Apply softmax to get final output
        y = self.softmax(y_linear)  # Shape: (m, o)

        return h_next, y

    def softmax(self, z):
        """
        Applies the softmax activation function.

        Args:
            z (ndarray): Raw output values (shape: (m, o)).

        Returns:
            ndarray: Softmax-activated output (shape: (m, o)).
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
