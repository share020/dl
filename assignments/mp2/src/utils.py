"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).

The convolution network should have a single hidden layer with multiple channels.

Due September 14 at 5:00 PM.

@author: Zhenye Na
@date: Sep 9
"""

import h5py


def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        onehot[i][y[i]] = 1
    return onehot


def accuracy(y_true, y_pred):
    # both are not one hot encoded
    return np.mean(y_pred == y_true)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def mnist_loader(path, one_hot=False):
    """
    Load MNIST Dataset

    inputs:

    outputs:

    """
    # load MNIST data
    MNIST_data = h5py.File(path, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
    MNIST_data.close()


    # reshape
    shape = (-1, 1, 28, 28)
    x_train = x_train.reshape(shape)
    x_test = x_test.reshape(shape)


    if one_hot:
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
