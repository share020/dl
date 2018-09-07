"""
Implementation of Neural Network from scratch in Python for the MNIST dataset.

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na
"""

import numpy as np
import random

from activations import *

class Network(object):

    def __init__(self, layers, lr, epochs, batch_size):
        """
        Initialize Neural Network model for MNIST
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.num_layers = len(layers)
        self.layers  = layers

        # Initialize weights and biases
        self.weights = [np.array([0])] + [np.random.randn(x, y) for y, x in zip(layers[1:], layers[:-1])]
        self.biases  = [np.random.randn(self.batch_size, y) for y in layers]

        # z's and a's
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases[1:]]

    def fit(self, x_train, y_train, validation_data=None):
        """
        Training process.
        """
        training_data = np.hstack((x_train, y_train))

        for epoch in xrange(self.epochs):
            mini_batches = [ training_data[k:k+self.batch_size] for k in xrange(0, len(training_data), self.batch_size) ]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]

                x, y = mini_batch[:,0:-10], mini_batch[:,-10:]
                self._forward_pass(x)
                delta_nabla_b, delta_nabla_w = self._back_prop(x, y)

                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.lr / self.batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.lr / self.batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]

                acc = self.metrics(y)
                print("Accuray {0}.".format(acc))

            print("Processed epoch {0}.".format(epoch))


    def _forward_pass(self, x):
        self._activations.insert(0, x)
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self._activations[i-1].dot(self.weights[i]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])
        self._prob = softmax(self._activations[-1])


    def _back_prop(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._prob - y) * sigmoid_prime(self._zs[-1])

        nabla_b[-1] = error
        nabla_w[-1] = self._activations[-2].transpose().dot(error)

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                error.dot(self.weights[l + 1].transpose()),
                sigmoid_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = self._activations[l - 1].transpose().dot(error)

        return nabla_b, nabla_w

    def metrics(self, y):
        """
        calculate accuracy
        """
        pass
        return (np.sum(np.argmax(self._activations[-1], axis=1) == y) / np.float(y.shape[0]))
