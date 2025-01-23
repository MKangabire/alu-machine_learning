#!/usr/bin/env python3
"""defines a neural network within one hidden layer"""

class NeuralNetwork:
    """defines a neural network"""

    def __init__(self, nx, nodes):
        """initializes"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(node, int):
            raise TypeError('nodes must be an integer')
        if node < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
