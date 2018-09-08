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
import skimage


from utils import *

parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('--dataroot', type=str, default="../MNISTdata.hdf5", help='path to dataset')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h', type=int, default=64, help='number of hidden units')
parser.add_argument('--beta', type=float, default=0.9, help='parameter for momentum')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

# parse the arguments
opt = parser.parse_args()


# load MNIST data
MNIST_data = h5py.File("../MNISTdata.hdf5", 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
x_test  = np.float32(MNIST_data['x_test'][:])
y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
MNIST_data.close()

np.random.seed(231)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-2)

solver = Solver(model, small_data,
                num_epochs=15, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()
