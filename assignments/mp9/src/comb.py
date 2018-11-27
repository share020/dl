"""
HW9: Human Action Recognition.

Part 3 - Sequence Model / Single Frame combination

@author: Zhenye Na
@credit: Logan Courtney
"""

import numpy as np
import os
import sys
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from helperFunctions import getUCF101
from helperFunctions import loadSequence
import resnet_3d

import h5py
import cv2
import argparse

from multiprocessing import Pool
