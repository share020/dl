"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na
"""

import time
import h5py
import argparse
import numpy as np

from mnist import *

parser = argparse.ArgumentParser()

# Data dir
parser.add_argument('--dataroot', type=str, default="../MNISTdata.hdf5", help='path to dataset')

# hyper-parameters
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--inputSize', type=int, default=784, help='input dimension')
parser.add_argument('--hiddenSize', type=int, default=256, help='the size of hidden units')
parser.add_argument('--outputSize', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')

# For debug
parser.add_argument('--debug', type=bool, default=False, help='debug')

# parse the arguments
opt = parser.parse_args()

def main():

    START_TIME = time.time()

    #load MNIST data
    MNIST_data = h5py.File(opt.dataroot, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0])).reshape(-1, 1)
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array( MNIST_data['y_test'][:,0])).reshape(-1, 1)
    MNIST_data.close()
    print('# Import duration ' + str(round((time.time() - START_TIME), 2)) + 's')
    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    nn = Network([opt.inputSize, opt.hiddenSize, opt.outputSize], opt.lr, opt.epochs, opt.batchSize)

    x_train = x_train[0:10000,:] if opt.debug else x_train
    y_train = y_train[0:10000,:] if opt.debug else y_train
    print("X_train shape: ", x_train.shape)
    print("y_test shape: ", y_test.shape)

    # x_test = x_test[0:256,:] if opt.debug else x_test
    # y_test = y_test[0:256,:] if opt.debug else y_test
    # print("x_test shape: ", x_test.shape)
    # print("y_test shape: ", y_test.shape)

    # train network using stochastic gradient descent
    nn.fit(zip(x_train, y_train))
    acc = nn.metrics(zip(x_test, y_test))
    print("Accuracy: ", acc)

if __name__ == '__main__':
    main()
