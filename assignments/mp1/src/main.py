"""
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).
Due: September 7.

@author: Zhenye Na
"""


import numpy as np

from mnist import *

def main():
    nn = nn()
    nn.train()
    nn.test()


if __name__ == '__main__':
    main()
