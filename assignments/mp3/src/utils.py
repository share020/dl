"""
HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset.

The convolution network should use
    (A) dropout
    (B) trained with RMSprop or ADAM, and
    (C) data augmentation. 

@author: Zhenye Na
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from cnn import *


def calculate_val_accuracy(testloader, is_gpu):
    """Util function to calculate val set accuracy.

    both overall and per class accuracy

    Args:
        testloader (torch.utils.data.DataLoader): val set
        is_gpu (bool): whether to run on GPU
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """

    TOTAL_CLASSES = 10

    correct = 0.
    total = 0.
    predictions = []

    class_correct = list(0. for i in range(TOTAL_CLASSES))
    class_total = list(0. for i in range(TOTAL_CLASSES))

    for data in testloader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)
        correct += (predicted == labels).sum()

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_accuracy = 100 * np.divide(class_correct, class_total)
    return 100 * correct / total, class_accuracy
