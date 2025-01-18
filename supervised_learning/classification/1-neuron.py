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

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias vector"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A