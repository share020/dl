"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using SGD.
It should achieve 97-98% accuracy on the Test Set.

@author: Zhenye Na
@date: Sep. 5
"""

from sklearn.metrics import classification_report
import numpy as np
import argparse
import h5py

from utils import *

parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h', type=int, default=64,
                    help='number of hidden units')
parser.add_argument('--beta', type=float, default=0.9,
                    help='parameter for momentum')
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size')

# parse the arguments
opt = parser.parse_args()


def main():
    """
    pipeline of training a neural network from scratch in Python for the MNIST dataset.
    """

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

    # number of batches
    batches = m // opt.batch_size

    # initialization
    params = {"W1": np.random.randn(opt.n_h, opt.n_x) * np.sqrt(1. / opt.n_x),
              "b1": np.zeros((opt.n_h, 1)) * np.sqrt(1. / opt.n_x),
              "W2": np.random.randn(digits, opt.n_h) * np.sqrt(1. / opt.n_h),
              "b2": np.zeros((digits, 1)) * np.sqrt(1. / opt.n_h)}

    dW1 = np.zeros(params["W1"].shape)
    db1 = np.zeros(params["b1"].shape)
    dW2 = np.zeros(params["W2"].shape)
    db2 = np.zeros(params["b2"].shape)


    # training
    for i in range(opt.epochs):

        # shuffle training set
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        for j in range(batches):

            # get mini-batch
            begin = j * opt.batch_size
            end = min(begin + opt.batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            # forward and backward
            cache = feed_forward(X, params)
            grads = back_propagate(X, Y, params, cache, m_batch)

            # with momentum (optional)
            dW1 = (opt.beta * dW1 + (1. - opt.beta) * grads["dW1"])
            db1 = (opt.beta * db1 + (1. - opt.beta) * grads["db1"])
            dW2 = (opt.beta * dW2 + (1. - opt.beta) * grads["dW2"])
            db2 = (opt.beta * db2 + (1. - opt.beta) * grads["db2"])

            # gradient descent
            params["W1"] = params["W1"] - opt.lr * dW1
            params["b1"] = params["b1"] - opt.lr * db1
            params["W2"] = params["W2"] - opt.lr * dW2
            params["b2"] = params["b2"] - opt.lr * db2

        # forward pass on training set
        cache = feed_forward(X_train, params)
        train_loss = compute_loss(Y_train, cache["A2"])

        # forward pass on test set
        cache = feed_forward(X_test, params)
        test_loss = compute_loss(Y_test, cache["A2"])
        print("Epoch {}: training loss = {}, test loss = {}".format(
            i + 1, train_loss, test_loss))


    # test accuracy using sklearn confusion matrix
    cache = feed_forward(X_test, params)
    predictions = np.argmax(cache["A2"], axis=0)
    labels = np.argmax(Y_test, axis=0)

    # get confusion matrix
    print(classification_report(predictions, labels))


if __name__ == '__main__':
    main()
