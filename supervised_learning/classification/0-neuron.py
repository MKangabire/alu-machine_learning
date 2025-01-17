#!/usr/bin/env python3
"""a neuron performing binary classification"""


import numpy as np


class Neuron:
    """class neuron"""
    def __init__(self, nx):
        """initializing method"""
        self.nx = nx
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
