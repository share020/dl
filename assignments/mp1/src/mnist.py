"""
Implementation of Neural Network from scratch in Python for the MNIST dataset.

@author: Zhenye Na
"""

import numpy as np
import random

from activations import *

class Network():
    """Neural Network."""

    def __init__(self, sizes, lr, epochs):
        """Initializde Neural Network for MNIST."""
        # # hyper-parameters
        # self.input_size  = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        self.epochs  = epochs
        self.lr = lr
        #
        # # weights
        # self.U = np.random.randn(input_size, hidden_size)
        # self.W = np.random.randn(hidden_size, output_size)
        # self.e = np.random.randn(hidden_size)
        # self.f = np.random.randn(output_size)
        self.num_layers = len(sizes)
        self.sizes   = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def fully_connected(self, X, U, e):
        """Fully Connected Layer.

        inputs:
            U: weight
            e: bias
        returns:
            X * U + e
        """
        return np.dot(X, U) + e

    def forward(self, X):
        '''
        forward propagation through the network.

        inputs:
            X: input data (batchSize, inputSize)
        returns:
            c: output (batchSize, outputSize)
        '''
        z2 = self.fully_connected(X, self.U, self.e)
        a2 = sigmoid(z2)

        z3 = self.fully_connected(a2, self.W, self.f)
        a4 = sigmoid(z3)
        return a4

    def SGD(self, x_train, y_train):
        """
        Train the neural network using SGD.
        The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.
        """
        print(x_train.shape)
        print(y_train.shape)
        training_data = np.hstack((x_train, y_train))
        n = len(training_data)

        for j in xrange(self.epochs):
            random.shuffle(training_data)
            for iter in xrange(n):
                if j % 10000 == 0:
                    self.lr /= 2

                self.update_mini_batch(training_data[iter])

            print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        delta_nabla_b, delta_nabla_w = self.backward(mini_batch[0: len(mini_batch) - 1], mini_batch[-1])
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (self.lr / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (self.lr / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backward(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = []           # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return (output_activations - y)
