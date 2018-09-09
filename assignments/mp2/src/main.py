"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""

import h5py
import argparse
import numpy as np


from cnn import *
from loss import *
from utils import *
from optim import *
from layers import *


parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('--dataroot', type=str, default="../MNISTdata.hdf5", help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
# parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
# parser.add_argument('--n_h', type=int, default=64, help='number of hidden units')
# parser.add_argument('--beta', type=float, default=0.9, help='parameter for momentum')
parser.add_argument('--minibatch_size', type=int, default=64, help='input batch size')


# filter parameters
parser.add_argument('--n_filter', type=int, default=32, help='number of filters')
parser.add_argument('--h_filter', type=int, default=3, help='height of filters')
parser.add_argument('--w_filter', type=int, default=3, help='width of filters')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--padding', type=int, default=1, help='zero padding')

# dataset parameters mnist_dims
parser.add_argument('--num_class', type=int, default=10, help='number of classes in MNIST dataset')


# mnist_dims, num_class=10, n_filter=opt.n_filter, h_filter=opt.h_filter, w_filter=opt.w_filter, stride=opt.stride, padding=opt.padding, loss_func=SoftmaxLoss


# parse the arguments
opt = parser.parse_args()


# def build_layers(X_dim, num_class):
#     conv = Conv(X_dim, n_filter=opt.n_filter, h_filter=opt.h_filter, w_filter=opt.w_filter, stride=opt.stride, padding=opt.padding)
#     relu_conv = ReLU()
#     flat = Flatten()
#     fc = FullyConnected(np.prod(conv.out_dim), num_class)
#     return [conv, relu_conv, flat, fc]



if __name__ == '__main__':
    """
    pipepline of convolution neural network from scratch in Python for the MNIST dataset
    """
    # load dataset
    training_set, test_set = mnist_loader(opt.dataroot)
    X, y = training_set
    X_test, y_test = test_set

    # mnist image dims
    mnist_dims = (1, 28, 28)

    # build layers
    # layers = build_layers(mnist_dims, num_class=10)
    # # create cnn class
    # cnn = CNN(layers)
    # cnn = sgd_momentum(cnn, X, y, minibatch_size=35, epoch=20,
    #                    learning_rate=0.01, X_test=X_test, y_test=y_test)

    # create cnn class
    cnn = CNN(mnist_dims,
              num_class=10,
              n_filter=opt.n_filter,
              h_filter=opt.h_filter,
              w_filter=opt.w_filter,
              stride=opt.stride,
              padding=opt.padding)

    # create GradientDescentOptimizer
    optim = GradientDescentOptimizer(cnn,
                                     X, y,
                                     minibatch_size=opt.minibatch_size,
                                     epochs=opt.epochs,
                                     learning_rate=opt.lr,
                                     X_test=X_test,
                                     y_test=y_test)

    # minimize loss
    optim.minimize()
