"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""


import numpy as np


class NeuralNetwork(object):
    """
    A three-layer convolutional network with the following architecture.

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    """
    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        Inputs:
            - input_dim: Tuple (C, H, W) giving size of input data
            - num_filters: Number of filters to use in the convolutional layer
            - filter_size: Size of filters to use in the convolutional layer
            - hidden_dim: Number of units to use in the fully-connected hidden layer
            - num_classes: Number of scores to produce from the final affine layer.
            - weight_scale: Scalar giving standard deviation for random initialization
              of weights.
            - reg: Scalar giving L2 regularization strength
            - dtype: numpy datatype to use for computation.

        Initialize weights and biases for the three-layer convolutional network.
        Weights should be initialized from a Gaussian with standard deviation equal to weight_scale.
        biases should be initialized to zero.
        All weights and biases should be stored in the dictionary self.params.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        # no. of channels, height and width
        C, H, W = input_dim

        # weights and biases for the convolutional layer
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # weights and biases of the hidden affine layer
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # weights and biases of the output affine layer
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        # Forward Pass: conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out1, cache1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, self.params['W2'], self.params['b2'])
        out3, cache3 = affine_forward(out2, self.params['W3'], self.params['b3'])
        scores = out3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dout = softmax_loss(out3, y)
        loss += self.reg * np.sum([np.sum(self.params['W%d' % i] ** 2) for i in [1, 2, 3]])
        dout, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
        grads['W3'] += 2 * self.reg * self.params['W3']
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache2)
        grads['W2'] += 2 * self.reg * self.params['W2']
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache1)
        grads['W1'] += 2 * self.reg * self.params['W1']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
