"""
Implementation of Neural Network from scratch in Python for the MNIST dataset.

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na


--------
this `Class` is not working, params never update.
will work on this later
--------

"""

import numpy as np
import random

from activations import *
from sklearn.metrics import classification_report


class Network(object):

    def __init__(self, input_size, hidden_size, lr, epochs, batch_size, output_size):
        """
        Initialize Neural Network model for MNIST
        """
        # hyperparameters
        self.n_x = input_size   # X_train.shape[0]
        self.n_h = hidden_size  # 64
        self.learning_rate = lr
        self.epochs = epochs
        self.beta = .9
        self.batch_size = batch_size  # 128
        self.batches = self.n_x // batch_size
        self.digits = output_size

        # initialization
        self.params = {"W1": np.random.randn(self.n_h, self.n_x),
                       "b1": np.zeros((self.n_h, 1)),
                       "W2": np.random.randn(self.digits, self.n_h),
                       "b2": np.zeros((self.digits, 1))}

        self.V_dW1 = np.zeros(self.params["W1"].shape)
        self.V_db1 = np.zeros(self.params["b1"].shape)
        self.V_dW2 = np.zeros(self.params["W2"].shape)
        self.V_db2 = np.zeros(self.params["b2"].shape)

    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def compute_loss(self, Y, Y_hat):

        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1. / m) * L_sum

        return L

    def feed_forward(self, X):

        cache = {}

        cache["Z1"] = np.matmul(self.params["W1"], X) + self.params["b1"]
        cache["A1"] = self.sigmoid(cache["Z1"])
        cache["Z2"] = np.matmul(
            self.params["W2"], cache["A1"]) + self.params["b2"]
        cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

        return cache

    def back_propagate(self, X, Y, cache):

        dZ2 = cache["A2"] - Y
        dW2 = (1. / self.m_batch) * np.matmul(dZ2, cache["A1"].T)
        db2 = (1. / self.m_batch) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.sigmoid(cache["Z1"]) * (1 - self.sigmoid(cache["Z1"]))
        dW1 = (1. / self.m_batch) * np.matmul(dZ1, X.T)
        db1 = (1. / self.m_batch) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads

    def train(self, X_train, Y_train, X_test, Y_test):

        # train
        for i in xrange(self.epochs):

            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            for j in xrange(self.batches):

                begin = j * self.batch_size
                end = min(begin + self.batch_size, X_train.shape[1] - 1)
                X = X_train_shuffled[:, begin:end]
                Y = Y_train_shuffled[:, begin:end]
                self.m_batch = end - begin

                cache = self.feed_forward(X)
                grads = self.back_propagate(X, Y, cache)

                self.V_dW1 = (self.beta * self.V_dW1 +
                              (1. - self.beta) * grads["dW1"])
                self.V_db1 = (self.beta * self.V_db1 +
                              (1. - self.beta) * grads["db1"])
                self.V_dW2 = (self.beta * self.V_dW2 +
                              (1. - self.beta) * grads["dW2"])
                self.V_db2 = (self.beta * self.V_db2 +
                              (1. - self.beta) * grads["db2"])

                # this is not working!!!! never update parameters
                self.params["W1"] -= self.learning_rate * self.V_dW1
                self.params["b1"] -= self.learning_rate * self.V_db1
                self.params["W2"] -= self.learning_rate * self.V_dW2
                self.params["b2"] -= self.learning_rate * self.V_db2

            cache = self.feed_forward(X_train)
            train_cost = self.compute_loss(Y_train, cache["A2"])
            cache = self.feed_forward(X_test)
            test_cost = self.compute_loss(Y_test, cache["A2"])
            print("Epoch {}: training cost = {}, test cost = {}".format(
                i + 1, train_cost, test_cost))

        print("Done.")

    def test(self, X_test, Y_test):
        """test"""
        cache = self.feed_forward(X_test)
        predictions = np.argmax(cache["A2"], axis=0)
        labels = np.argmax(Y_test, axis=0)

        print(classification_report(predictions, labels))
