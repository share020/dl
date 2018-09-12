"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""

import h5py
import numpy as np


def mnist_loader(path, one_hot=False, debug=True):
    """
    Load MNIST Dataset.

    inputs:
        path: str, path to dataset
    outputs:
        (x_train, y_train), (x_test, y_test): training set, test set
    """
    # load MNIST data
    print(">>> Loading MNIST dataset...")
    MNIST_data = h5py.File(path, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
    MNIST_data.close()

    # reshape to input shape
    shape = (-1, 1, 28, 28)
    x_train = x_train.reshape(shape)
    x_test = x_test.reshape(shape)

    if one_hot:
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)

    if debug:
        num_training, num_test = 20480, 10000
        x_train, y_train = x_train[range(
            num_training)], y_train[range(num_training)]
        x_test, y_test = x_test[range(num_test)], y_test[range(num_test)]

    return (x_train, y_train), (x_test, y_test)


def one_hot_encode(y, num_class):
    """
    One-Hot Encoding

    inputs:
        y: ground truth label
        num_class: number of classes

    return:
        onehot: one-hot encoded labels
    """
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        onehot[i][y[i]] = 1
    return onehot


def softmax(x):
    """
    Perform Softmax

    return:
        np.exp(x) / np.sum(np.exp(x))
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'),
                  field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    """
    An implementation of im2col based on some fancy indexing
    """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(
        x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    An implementation of col2im based on fancy indexing and np.add.at
    """
    N, C, H, W = x_shape

    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(
        x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
