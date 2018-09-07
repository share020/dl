import numpy as np

def sigmoid(z):
    """
    sigmoid activation function.

    inputs: z
    outputs: sigmoid(z)
    """
    s = 1. / (1. + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L
