"""
HW4: Implement a deep residual neural network for CIFAR100.

Part-1: Build the Residual Network

Due October 5 at 5:00 PM.

@author: Zhenye Na
"""

import torch
import torch.nn as nn


def initialize_weights(module):
    """Initialize weights."""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock(nn.Module):
    """Basic Block of ReseNet."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """Basic Block of ReseNet Builder."""
        super(BasicBlock, self).__init__()

        # First conv3x3 layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)

        #  Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # ReLU Activation Function
        self.relu = nn.ReLU(inplace=True)

        # Second conv3x3 layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        #  Batch Normalization
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # self.shortcut = nn.Sequential()
        # if in_channels != out_channels:
        #     self.shortcut.add_module(
        #         'conv',
        #         nn.Conv2d(
        #             in_channels,
        #             out_channels,
        #             kernel_size=1,
        #             stride=stride,  # downsample
        #             padding=0,
        #             bias=False))
        #     self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN


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



class ResNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self, block, layers, num_classes=100):
        """Residual Neural Network Builder."""
        super(ResNet, self).__init__()

        self.inplanes = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.01, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)

        # blocks of conv layers
        self.block1 = self._make_block(block, layers[0], planes=32)
        self.block2 = self._make_block(block, layers[1], planes=64, stride=2)
        self.block3 = self._make_block(block, layers[2], planes=128, stride=2)
        self.block4 = self._make_block(block, layers[3], planes=256, stride=2)

        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # # initialize weights
        # self.apply(initialize_weights)


    def _make_block(self, block, configs, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, configs):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)




    def forward(self, x):
        """Forward pass of ResNet."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





def resnet_cifar(**kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model