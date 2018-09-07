"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na
"""


import argparse
import numpy as np

from mnist import *
from mnist_loader import *

parser = argparse.ArgumentParser()

# hyper-parameters
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--inputSize', type=int, default=784, help='input dimension')
parser.add_argument('--hiddenSize', type=int, default=256, help='the size of hidden units')
parser.add_argument('--outputSize', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')

# For debug
parser.add_argument('--debug', type=bool, default=False, help='debug')

# parse the arguments
opt = parser.parse_args()

def main():

    x_train, y_train, x_test, y_test = load_data()
    # x_train = x_train[0:10000,:] if opt.debug else x_train
    # y_train = y_train[0:10000,:] if opt.debug else y_train
    # print("X_train shape: ", x_train.shape)
    # print("y_test shape: ", y_test.shape)


    nn = Network([opt.inputSize, opt.hiddenSize, opt.outputSize],
                  opt.lr,
                  opt.epochs,
                  opt.batchSize)


    # x_test = x_test[0:256,:] if opt.debug else x_test
    # y_test = y_test[0:256,:] if opt.debug else y_test
    # print("x_test shape: ", x_test.shape)
    # print("y_test shape: ", y_test.shape)

    # train network using stochastic gradient descent
    nn.fit(x_train, y_train)
    acc = nn.metrics(y_test)
    print("Accuracy: ", acc)

if __name__ == '__main__':
    main()
