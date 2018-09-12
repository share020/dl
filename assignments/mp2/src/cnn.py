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
            mnist_dims: MNIST image dimension (1 * 28 * 28)
            num_class: number of classes in MNIST dataset. e.g. 10
            n_filter: number of filters
            h_filter: height of filters
            w_filter: width of filters
            stride: stride
            padding: padding
            loss_func: loss function
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

        inputs:
            mnist_dims: MNIST image dimension (1 * 28 * 28)
            n_filter: number of filters
            h_filter: height of filters
            w_filter: width of filters
            stride: stride
            padding: padding
            num_class: number of classes in MNIST dataset. e.g. 10

        return:
            list: a list of layers in CNN model
        """
        # Convolutional layer
        conv = Conv(mnist_dims, n_filter, h_filter, w_filter, stride, padding)

        # ReLU activation layer
        relu_conv = ReLU()

        # flatten parameters
        flat = Flatten()

        # Fully Connected layer
        fc = FullyConnected(np.prod(conv.out_dim), num_class)

        return [conv, relu_conv, flat, fc]

    def forward(self, X):
        """
        Perform forward pass in each layer of CNN model

        inputs:
            X: training / test images

        return:
            X: result of fprward pass
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        """
        Perform back propagation in each layer of CNN model
        """
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):
        """
        Train.

        inputs:
            X: training / test images
            y: training / test labels

        return:
            loss: training loss
            grads: parameters' gradients
        """
        # forward pass
        out = self.forward(X)

        # compute loss
        loss, dout = self.loss_func(out, y)

        # back propagation
        grads = self.backward(dout)

        return loss, grads

    def predict(self, X):
        """
        Predict

        inputs:
            X: training / test images
            y: training / test labels

        return:
            predicted labels
        """
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)
