"""
Implementation of Neural Network from scratch in Python for the MNIST dataset.

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
        self.input_size = layers[0]
        self.num_layers = len(layers)
        self.layers  = layers
        self.biases  = [np.random.randn(self.batch_size, x) for x in layers[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layers[:-1], layers[1:])]

        for w in self.weights:
            print("w: ", w.shape)

        for b in self.biases:
            print("b: ", b.shape)

    def feedforward(self, a):
        """
        Return the output of the network if ``a`` is input.
        """
        for b, w in zip(self.biases, self.weights):
            print("w", w.shape)
            print("a", a.shape)
            print("b", b.shape)
            a = sigmoid(np.dot(a, w) + b)
            i = 0
            print(i)
            i += 1
        return a

    def SGD(self, training_data, test_data=None):
        """
        Train the neural network using SGD.

        """
        N = len(training_data)

        for j in xrange(self.epochs):

            if j % 10 == 0:
                print("Epoch: ", j)

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.batch_size]
                for k in xrange(0, N - self.batch_size - 1, self.batch_size)]
            # mini_batches = np.copy(training_data)

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            # if test_data is not None:
            #     print("Epoch {0}: {1} / {2}".format(
            #         j, self.evaluate(test_data), n_test))
            # else:
            print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        x, y = np.array(mini_batch[:,0:-1]).reshape(-1, self.input_size), np.array(mini_batch[:,-1]).reshape(-1, 1)
        print("x shape", x.shape)
        print("y shape", y.shape)


        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (self.lr / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (self.lr / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation  = x
        activations = [x] # list to store all the activations, layer by layer
        zs = []           # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.d_loss_o(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2])


        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1], delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        calculate accuracy
        """
        # print(test_data.shape)
        # test_results = np.argmax(self.feedforward(test_data[:,0:-1]))
        # return np.sum(np.equal(test_results, test_data[:,-1]))
        N = len(test_data)
        mini_batches = [
            test_data[k:k+self.batch_size]
            for k in xrange(0, N - self.batch_size - 1, self.batch_size)]

        acc = 0
        for mini_batch in mini_batches:
            acc += np.sum(np.equal(np.argmax(self.feedforward(test_data[:,0:-1])), test_data[:,-1]))

        return acc

    def d_loss_o(self, o, gt):
        """
        computes the derivative of the L2 loss with respect to
        the network's output.
        """
        return (o - gt)

    def predict(self, test_data):
        n_test = len(test_data)
        print("Epoch {0}: {1} / {2}".format(
            j, self.evaluate(test_data)))
