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
        self.weights = [np.array([0])] + [np.random.randn(x, y) for y, x in zip(layers[1:], layers[:-1])]

        # self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.biases = [np.random.randn(1, y) for y in layers]

        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

    def fit(self, training_data, validation_data=None):
        """
        Training process.
        """
        for epoch in xrange(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.batch_size] for k in
                xrange(0, len(training_data), self.batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                # x, y = mini_batch[:,0:-1], mini_batch[:,-1]
                    x = x.reshape(1,-1)
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.lr / self.batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.lr / self.batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]

            print("Processed epoch {0}.".format(epoch))
            acc = self.metrics(training_data)
            print("Accuray {0}.".format(acc))


    def predict(self, x):
        """
        prediction
        """

        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self, x):
        self._activations[0] = x
        for i in xrange(1, self.num_layers):
            # self._activations[i-1] = np.array(self._activations[i-1])
            self._zs[i] = (
                self._activations[i-1].dot(self.weights[i]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])

    def _back_prop(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * sigmoid_prime(self._zs[-1])
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


    def metrics(self, test_data):
        """
        calculate accuracy
        """
        mini_batches = [
            test_data[k:k + self.batch_size] for k in
            xrange(0, len(test_data), self.batch_size)]
        print(len(mini_batches))

        n = 0.
        acc = 0.
        for mini_batch in mini_batches:
            for x, y in mini_batch:
                n += 1
                if  y == self.predict(x):
                    acc += 1

        return (acc / n)
