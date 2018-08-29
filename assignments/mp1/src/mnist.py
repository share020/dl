"""
Implementation of Neural Network from scratch in Python for the MNIST dataset

@author: Zhenye Na
"""


import numpy as np



class nn():
    """Neural Network"""
    def __init__(inputSize, hiddenSize, outputSize):
        # weights
        self.U = np.random.randn(inputSize, hiddenSize)
        self.W = np.random.randn(hiddenSize, outputSize)
        self.e = np.random.randn(hiddenSize)
        self.f = np.random.randn(outputSize)
