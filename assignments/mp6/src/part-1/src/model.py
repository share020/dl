"""
HW6: Understanding CNNs and Generative Adversarial Networks.

@author: Zhenye Na
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self):
        """
        Discriminator Builder.

        Layer Normalization after every convolution operation followed by a Leaky ReLU
        """
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[32,32]),
            nn.LeakyReLU(),

            # conv2
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm(normalized_shape=[16,16]),
            nn.LeakyReLU(),

            # conv3
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[16,16]),
            nn.LeakyReLU(),

            # conv4
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm(normalized_shape=[8,8]),
            nn.LeakyReLU(),

            # conv5
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[8,8]),
            nn.LeakyReLU(),

            # conv6
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[8,8]),
            nn.LeakyReLU(),

            # conv7
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[8,8]),
            nn.LeakyReLU(),

            # conv8
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm(normalized_shape=[4,4]),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        )

        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)


    def forward(self, x):
        """
        Forward pass of discriminator.

        Args:
            x: input images / input data

        Returns:
            out1: output of fc1
            out2: output of fc10
        """
        out = self.conv(x)
        out = out.view(out.size(0), -1)

        # critic
        out1 = self.fc1(out)

        # auxiliary classifier
        out2 = self.fc10(out)

        return out1, out2



class Generator(nn.Module):
    """Generator."""

    def __init__(self):
        """Generator Builder."""
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(128, 196 * 4 * 4)

        self.tconv = nn.Sequential(
            # conv1
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv2
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv3
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv4
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv5
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv6
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv7
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

            # conv8
            nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        # self.conv1_block = self._tconv_bn_relu_block(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=0)
        # self.conv2_block = self._tconv_bn_relu_block(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        # self.conv3_block = self._tconv_bn_relu_block(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        # self.conv4_block = self._tconv_bn_relu_block(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        # self.conv5_block = self._tconv_bn_relu_block(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=0)
        # self.conv6_block = self._tconv_bn_relu_block(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        """Forward pass of Genrator."""
        out = self.fc1(x)
        out = F.tanh(self.tconv(out))

        return out
