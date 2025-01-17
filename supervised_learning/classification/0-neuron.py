import numpy as np

class Neuron:
    def __init__(self, nx):
        self.nx = nx
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
