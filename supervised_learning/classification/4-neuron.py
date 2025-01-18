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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        div = 1.0000001 - A
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(div))
        return cost

    def evaluate(self, X, Y):
        """Evaluates neuron's prediction"""
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost
