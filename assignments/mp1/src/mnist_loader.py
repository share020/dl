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

    return x_train, y_train, x_test, y_test
