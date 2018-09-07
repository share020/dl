"""
Implementation of Neural Network from scratch in Python for the MNIST dataset.
Helper module to provide activation to network layers.

@author: Zhenye Na

activations with their derivates are available:
    - Sigmoid
    - Softmax
    - ReLU
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid activation function.

    inputs: z
    outputs: sigmoid(z)
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    derivative of sigmoid.

    inputs:
        s = sigmoid(x)
    outputs:
        derivative sigmoid(x) as a function of s
    """
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    """
    Softmax activation function.

    inputs: z
    outputs: exp(z) / sum(exp(z))
    """
    return np.exp(z) / np.sum(np.exp(z))


def softmax_prime(z):
    """
    derivative of softmax.

    inputs:
        exp(z) / sum(exp(z))
    outputs:
        derivative of exp(z) / sum(exp(z))
    """
    return softmax(z) * (1 - softmax(z))


def relu(z):
    """
    ReLU activation function.

    inputs: z
    outputs: z if z > 0 else 0
    """
    return np.maximum(z, 0)


def relu_prime(z):
    """
    ReLU activation function.

    inputs: z = np.maximum(z, 0)
    outputs: 1 if z > 0 else 0
    """
    return float(z > 0)
