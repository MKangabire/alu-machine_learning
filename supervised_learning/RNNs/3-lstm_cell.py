#!/usr/bin/env python3
"""recurrent neural networks"""


import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit for one time step.
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Forget gate parameters
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # Update/input gate parameters
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # Candidate cell state parameters
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # Output gate parameters
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        # Output layer parameters
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (ndarray): Previous hidden state of shape (m, h)
            c_prev (ndarray): Previous cell state of shape (m, h)
            x_t (ndarray): Current input data of shape (m, i)

        Returns:
            h_next (ndarray): Next hidden state
            c_next (ndarray): Next cell state
            y (ndarray): Output of the cell
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)

        # Input gate
        u_t = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)

        # Candidate cell state
        c_hat = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # New cell state
        c_next = f_t * c_prev + u_t * c_hat

        # Output gate
        o_t = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        # New hidden state
        h_next = o_t * np.tanh(c_next)

        # Output of the cell
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
