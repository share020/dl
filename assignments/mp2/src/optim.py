def sgd(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, verbose=True, X_test=None, y_test=None):
    minibatches = get_minibatches(X_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        if verbose:
            print("Epoch {0}".format(i + 1))
        for X_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(X_mini, y_mini)
            vanilla_update(nnet.params, grads, learning_rate=learning_rate)
        if verbose:
            train_acc = accuracy(y_train, nnet.predict(X_train))
            test_acc = accuracy(y_test, nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(
                loss, train_acc, test_acc))
    return nnet

import numpy as np
from sklearn.utils import shuffle

from cnn import *
from utils import *
from layers import *
from loss import *


class GradientDescentOptimizer(object):
    """docstring for SGDOptimizer."""
    def __init__(self, nnet, X_train, y_train, minibatch_size, epochs, learning_rate, verbose=True, X_test=None, y_test=None):
        self.nnet = nnet

        self.X_train = X_train
        self.y_train = y_train

        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # print logs
        self.verbose = verbose

        # for test
        self.X_test = X_test
        self.y_test = y_test

    def minimize(self):
        """
        minimize loss
        """

        minibatches = self.get_minibatches()

        for i in range(self.epochs):
            loss = 0

            for X_mini, y_mini in minibatches:
                loss, grads = self.nnet.train_step(X_mini, y_mini)
                self.vanilla_update(grads)

            if self.verbose:
                # print("Epoch {0}".format(i + 1))
                train_acc = accuracy(self.y_train, self.nnet.predict(self.X_train))
                test_acc = accuracy(self.y_test, self.nnet.predict(self.X_test))
                print("Epoch {0} | Loss = {1} | Training Accuracy = {2} | Test Accuracy = {3}".format(i + 1, loss, train_acc, test_acc))

    def get_minibatches(self, isShuffle=True):
        m = self.X_train.shape[0]
        minibatches = []

        print(self.X_train.shape)
        print(self.y_train.shape)

        X, y = self.X_train, self.y_train

        if isShuffle:
            X, y = shuffle(X, y)

        for i in range(0, m, self.minibatch_size):
            X_batch = X[i:i + self.minibatch_size, :, :, :]
            y_batch = y[i:i + self.minibatch_size, ]
            minibatches.append((X_batch, y_batch))
        return minibatches

    def vanilla_update(self, grads):
        """
        update parameters
        """
        for param, grad in zip(self.nnet.params, reversed(grads)):
            for i in range(len(grad)):
                param[i] += - self.learning_rate * grad[i]
