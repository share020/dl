import h5py
import numpy as np


def load_data():

    # load MNIST data
    MNIST_data = h5py.File("../MNISTdata.hdf5", 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0])).reshape(-1, 1)
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array( MNIST_data['y_test'][:,0])).reshape(-1, 1)
    MNIST_data.close()

    # mnist_data = np.hstack((x_train, y_train))
    # np.random.shuffle(mnist_data)
    # x_train = mnist_data[:,0:-1]
    # y_train = mnist_data[:,-1].reshape(-1, 1)
    y_train = one_hot_encoding(y_train)

    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    return x_train, y_train, x_test, y_test


def one_hot_encoding(vector):
    result = np.zeros((len(vector), 10))
    result[np.arange(len(vector)), np.int32(vector)] = 1
    return result
