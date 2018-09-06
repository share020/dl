"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).
Due: September 7.

@author: Zhenye Na
"""

import time
import h5py
import argparse
import numpy as np

from mnist import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', type=str, default="../MNISTdata.hdf5", help='path to dataset')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--inputSize', type=int, default=784, help='input dimension')
parser.add_argument('--hiddenSize', type=int, default=256, help='the size of hidden units')
parser.add_argument('--outputSize', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')

opt = parser.parse_args()


def main():

    START_TIME = time.time()

    #load MNIST data
    MNIST_data = h5py.File(opt.dataroot, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0])).reshape(len(MNIST_data['y_train'][:,0]), 1)
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array( MNIST_data['y_test'][:,0])).reshape(len(MNIST_data['y_test'][:,0]), 1)
    MNIST_data.close()
    print('>>> Import duration ' + str(round((time.time() - START_TIME), 2)) + 's')


    nn = Network([opt.inputSize, opt.hiddenSize, opt.outputSize], opt.lr, opt.epochs, opt.batchSize)

    # train network using stochastic gradient descent
    nn.SGD(np.hstack((x_train, y_train)), np.hstack((x_test, y_test)))
    acc = nn.evaluate(np.hstack((x_test, y_test)))
    print("Accuracy: ", acc)

if __name__ == '__main__':
    main()
