"""
HW4: Implement a deep residual neural network for CIFAR100.

Use data augmentation techniques and dropout.

Due October 5 at 5:00 PM.

@author: Zhenye Na
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block of ReseNet."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """Basic Block of ReseNet Builder."""
        super(BasicBlock, self).__init__()

        # First conv3x3 layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)

        #  Batch Normalization
        self.bn1 = nn.BatchNorm2d(out_channels)

        # ReLU Activation Function
        self.relu = nn.ReLU(inplace=True)

        # Second conv3x3 layer
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        #  Batch Normalization
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample if needed
        self.downsample = downsample


    def forward(self, x):
        """Forward Pass of Basic Block."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out



