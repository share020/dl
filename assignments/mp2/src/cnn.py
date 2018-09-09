"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""


import numpy as np

from loss import *
from utils import *
from layers import *


class CNN:
    """
    Convolutional Neural Network model
    """
    def __init__(self, mnist_dims, num_class, n_filter, h_filter,
                 w_filter, stride, padding, loss_func=SoftmaxLoss):
        """
        Initialize a CNN model

        inputs:
            mnist_dims:
            num_class:
            n_filter:
            h_filter:
            w_filter:
            stride:
            padding:
            loss_func:
        """
        # build layers
        self.layers = self.build_layers(mnist_dims,
                                        n_filter,
                                        h_filter,
                                        w_filter,
                                        stride,
                                        padding,
                                        num_class)

        # parameters
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)

        # loss function
        self.loss_func = loss_func

    def build_layers(self, mnist_dims, n_filter, h_filter,
                     w_filter, stride, padding, num_class):
        """
        Build layers of CNN model
        """
        # Convolutional layer
        conv = Conv(mnist_dims, n_filter, h_filter, w_filter, stride, padding)

        # ReLU activation layer
        relu_conv = ReLU()

        # flatten
        flat = Flatten()

        # Fully Connected layer
        fc = FullyConnected(np.prod(conv.out_dim), num_class)

        return [conv, relu_conv, flat, fc]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):
        """
        """
        # forward pass
        out = self.forward(X)

        # compute loss
        loss, dout = self.loss_func(out, y)

        # back propagation
        grads = self.backward(dout)

        return loss, grads

    def predict(self, X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)
