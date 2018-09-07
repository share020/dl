"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na
"""

from sklearn.metrics import classification_report
import numpy as np
import argparse
import h5py

from utils import *

# generate seed for the same output as report
np.random.seed(138)


# load MNIST data
MNIST_data = h5py.File("../MNISTdata.hdf5", 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
x_test  = np.float32(MNIST_data['x_test'][:])
y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
MNIST_data.close()


# stack together for next step
X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))


# one-hot encoding
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)


# number of training set
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]


# shuffle training set
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h', type=int, default=64,
                    help='number of hidden units')
parser.add_argument('--beta', type=float, default=0.9,
                    help='number of hidden units')
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size')

# parse the arguments
opt = parser.parse_args()

# number of batches
batches = m // opt.batch_size

# initialize weights and biases
params = {"W1": np.random.randn(opt.n_h, opt.n_x) * np.sqrt(1. / opt.n_x),
          "b1": np.zeros((opt.n_h, 1)) * np.sqrt(1. / opt.n_x),
          "W2": np.random.randn(digits, opt.n_h) * np.sqrt(1. / opt.n_h),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / opt.n_h)}

# derivative of corresponding parameters
V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)


def feed_forward(X, params):
    """
    feed forward network (fully connected layers)

    inputs:
        params: dictionay a dictionary contains all the weights and biases

    return:
        cache: dictionay a dictionary contains all the fully connected units and activations
    """
    cache = {}

    # Z1 = W1.dot(x) + b1
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]

    # A1 = sigmoid(Z1)
    cache["A1"] = sigmoid(cache["Z1"])

    # Z2 = W2.dot(A1) + b2
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    # A2 = softmax(Z2)
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache


def back_propagate(X, Y, params, cache):
    """
    back propagation

    inputs:
        params: dictionay a dictionary contains all the weights and biases
        cache: dictionay a dictionary contains all the fully connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """
    # error at last layer
    dZ2 = cache["A2"] - Y

    # gradients at last layer (Py2 need 1. to transform to float)
    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # back propgate at first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


# training
for i in range(opt.epochs):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * opt.batch_size
        end = min(begin + opt.batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        # forward and backward
        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache)

        # with momentum (optional)
        V_dW1 = (opt.beta * V_dW1 + (1. - opt.beta) * grads["dW1"])
        V_db1 = (opt.beta * V_db1 + (1. - opt.beta) * grads["db1"])
        V_dW2 = (opt.beta * V_dW2 + (1. - opt.beta) * grads["dW2"])
        V_db2 = (opt.beta * V_db2 + (1. - opt.beta) * grads["db2"])

        # gradient descent
        params["W1"] = params["W1"] - opt.lr * V_dW1
        params["b1"] = params["b1"] - opt.lr * V_db1
        params["W2"] = params["W2"] - opt.lr * V_dW2
        params["b2"] = params["b2"] - opt.lr * V_db2

    # forward pass on training set
    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])

    # forward pass on test set
    cache = feed_forward(X_test, params)
    test_cost = compute_loss(Y_test, cache["A2"])
    print("Epoch {}: training cost = {}, test cost = {}".format(
        i + 1, train_cost, test_cost))


# test accuracy using sklearn confusion matrix
cache = feed_forward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

# get confusion matrix
print(classification_report(predictions, labels))
