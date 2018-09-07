"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na
"""

import argparse
import numpy as np

from mnist import *
from utils import *
from mnist_loader import *

parser = argparse.ArgumentParser()

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--inputSize', type=int, default=784, help='input dimension')
parser.add_argument('--hiddenSize', type=int, default=256, help='the size of hidden units')
parser.add_argument('--outputSize', type=int, default=10, help='the size of output')

# parse the arguments
opt = parser.parse_args()

def main():

    X_train, y_train, X_test, y_test = load_data()

    nn = NeuralNetwork(n_output=10,
                       n_features=opt.inputSize,
                  n_hidden=opt.hiddenSize,
                  l2=0.1,
                  epochs=opt.epochs,
                  learning_rate=opt.lr,
                  decay_rate=0.001,
                  minibatch_size=opt.batchSize)

    # train network using stochastic gradient descent
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)
    test_acc = np.sum(y_pred == y_test) / y_pred.shape[0]

if __name__ == '__main__':
    main()
