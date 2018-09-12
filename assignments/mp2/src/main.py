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
parser.add_argument('--dataroot', type=str,
                    default="../MNISTdata.hdf5", help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--minibatch_size', type=int,
                    default=64, help='input batch size')


# filter parameters
parser.add_argument('--n_filter', type=int, default=32,
                    help='number of filters')
parser.add_argument('--h_filter', type=int, default=7,
                    help='height of filters')
parser.add_argument('--w_filter', type=int, default=7, help='width of filters')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--padding', type=int, default=1, help='zero padding')


# dataset parameters mnist_dims
parser.add_argument('--num_class', type=int, default=10,
                    help='number of classes in MNIST dataset')
parser.add_argument('--num_channel', type=int, default=1,
                    help='number of channels in MNIST dataset')
parser.add_argument('--img_height', type=int, default=28,
                    help='height of images in MNIST dataset')
parser.add_argument('--img_width', type=int, default=28,
                    help='width of images in MNIST dataset')


# parse the arguments
opt = parser.parse_args()


if __name__ == '__main__':
    """
    pipepline of convolution neural network from scratch in Python for the MNIST dataset
    """
    # load dataset
    training_set, test_set = mnist_loader(
        opt.dataroot, one_hot=False, debug=True)
    X, y = training_set
    X_test, y_test = test_set

    # mnist image dims
    mnist_dims = (opt.num_channel, opt.img_height, opt.img_width)

    print(">>> Initialize CNN model ...")

    # create cnn class
    cnn = CNN(mnist_dims,
              num_class=opt.num_class,
              n_filter=opt.n_filter,
              h_filter=opt.h_filter,
              w_filter=opt.w_filter,
              stride=opt.stride,
              padding=opt.padding)

    print(">>> Initialize GradientDescentOptimizer ...")

    # create GradientDescentOptimizer
    optim = GradientDescentOptimizer(cnn,
                                     X, y,
                                     minibatch_size=opt.minibatch_size,
                                     epochs=opt.epochs,
                                     learning_rate=opt.lr,
                                     X_test=X_test,
                                     y_test=y_test)

    print(">>> Training ...")

    # minimize loss
    optim.minimize()
