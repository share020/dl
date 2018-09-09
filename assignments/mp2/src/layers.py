"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""

import numpy as np

from utils import *


class Conv():

    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride, padding):
        """
        Initialize a Convolutional layer
        """
        # inpput dimension
        self.d_X, self.h_X, self.w_X = X_dim

        # filter dimension
        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter

        # stride and padding
        self.stride, self.padding = stride, padding

        # initialize paprameters
        self.W = np.random.randn(n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]

        # spatial size of output
        self.h_out = (self.h_X - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_filter + 2 * padding) / stride + 1

        # if not self.h_out.is_integer() or not self.w_out.is_integer():
        #     raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)

        # output dimension
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, X):
        """
        forward pass of conv layer
        """
        self.n_X = X.shape[0]

        self.X_col = im2col_indices(X, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        W_row = self.W.reshape(self.n_filter, -1)

        out = W_row.dot(self.X_col) + self.b
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """
        """
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dW = dout_flat.dot(self.X_col.T)
        dW = dW.reshape(self.W.shape)
        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.reshape(self.n_filter, -1)

        dX_col = W_flat.T.dot(dout_flat)
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.h_filter, self.w_filter, self.padding, self.stride)

        return dX, [dW, db]


class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        out = dout.reshape(self.X_shape)
        return out, ()


class FullyConnected():

    def __init__(self, in_size, out_size):
        """
        Initialize a Fully Connected Layer
        """
        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = self.X.dot(self.W) + self.b
        return out

    def backward(self, dout):
        dW = self.X.T.dot(dout)
        db = np.sum(dout, axis=0)
        dX = dout.dot(self.W.T)
        return dX, [dW, db]


class ReLU():
    def __init__(self):
        """
        Initialize a ReLU Layer
        """
        self.params = []

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []


class sigmoid():
    def __init__(self):
        """
        Initialize a Sigmoid Layer
        """
        self.params = []

    def forward(self, X):
        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * self.out * (1 - self.out)
        return dX, []
