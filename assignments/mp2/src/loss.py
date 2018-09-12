import numpy as np
from utils import softmax
from layers import Conv, FullyConnected


def SoftmaxLoss(X, y):
    """
    Softmax loss
    """
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    dx = p
    dx[range(m), y] -= 1
    dx /= m
    return loss, dx
