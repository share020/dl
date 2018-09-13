"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""


import numpy as np
from sklearn.utils import shuffle

from cnn import *
from loss import *
from utils import *
from layers import *


class GradientDescentOptimizer(object):
    """Stochastic Gradient Descent Optimizer."""

    def __init__(self, nnet, X_train, y_train, minibatch_size, epochs,
                 learning_rate, verbose=True, X_test=None, y_test=None):
        """
        Stochastic Gradient Descent Optimizer.

        inputs
            nnet: CNN model
            X_train: training images
            y_train: training labels (ground truth)
            minibatch_size: size of mini-batches
            epochs: training epochs
            learning_rate: learning rate
            verbose: whether print logs
            X_test: test images
            y_test: test labels (ground truth)
        """
        # neural net model
        self.nnet = nnet

        # training set
        self.X_train = X_train
        self.y_train = y_train

        # hyper-parameters
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.learning_rate  = learning_rate

        # print logs
        self.verbose = verbose

        # for test
        self.X_test = X_test
        self.y_test = y_test

    def minimize(self):
        """
        Optimizer minimize step
        """
        # permute minibatches
        minibatches = self.get_minibatches()

        for i in range(self.epochs):

            # training and update params
            for X_mini, y_mini in minibatches:
                loss, grads = self.nnet.train_step(X_mini, y_mini)
                self.update_params(grads)

            if self.verbose:
                train_acc = self.accuracy(
                    self.y_train, self.nnet.predict(self.X_train))
                test_acc = self.accuracy(
                    self.y_test, self.nnet.predict(self.X_test))
                print("Epoch {0}, Loss = {1}, Training Accuracy = {2}, Test Accuracy = {3}".format(
                    i + 1, loss, train_acc, test_acc))

    def get_minibatches(self, isShuffle=True):
        """
        get mini batches

        inputs:
            isShuffle: bool whether shuffle dataset

        return:
            minibatches: list list of tuple of minibatches of dataset
        """
        m = self.X_train.shape[0]
        minibatches = []

        # make a copy of X_train and y_train
        X, y = self.X_train.copy(), self.y_train.copy()

        # shuffle data
        if isShuffle:
            X, y = shuffle(X, y)

        for i in range(0, m, self.minibatch_size):
            X_batch = X[i:i + self.minibatch_size, :, :, :]
            y_batch = y[i:i + self.minibatch_size, ]
            minibatches.append((X_batch, y_batch))

        return minibatches

    def update_params(self, grads):
        """
        Perform weights and biases update

        inputs:
            grads: gradients of weights and biases
        """
        for param, grad in zip(self.nnet.params, reversed(grads)):
            for i in range(len(grad)):
                param[i] += - self.learning_rate * grad[i]

    def accuracy(self, y_true, y_pred):
        """
        Calculate accuracy

        inputs:
            y_true: grounf truth labels
            y_pred: prediction of labels
        """
        return np.mean(y_pred == y_true)
