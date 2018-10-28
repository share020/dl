"""
HW6: Understanding CNNs and Generative Adversarial Networks.

Part 2: Visualization

@author: Zhenye Na
"""

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
            nn.LayerNorm(normalized_shape=[32, 32]),
            nn.LeakyReLU(),

            # conv2
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm(normalized_shape=[16, 16]),
            nn.LeakyReLU(),

            # conv3
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[16, 16]),
            nn.LeakyReLU(),

            # conv4
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm(normalized_shape=[8, 8]),
            nn.LeakyReLU(),

            # conv5
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[8, 8]),
            nn.LeakyReLU(),

            # conv6
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[8, 8]),
            nn.LeakyReLU(),

            # conv7
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm(normalized_shape=[8, 8]),
            nn.LeakyReLU(),

            # conv8
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm(normalized_shape=[4, 4]),
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


class DiscriminatorBottom(nn.Module):
    """Discriminator Bottom layers."""

    def __init__(self, model):
        """Discriminator Bottom layers Builder."""
        super(DiscriminatorBottom, self).__init__()
        # 4th conv blocks
        self.features = nn.Sequential(*(list(model.modules())[2][:10]))

    def forward(self, x):
        """Forward pass."""
        h = self.features(x)
        h = F.max_pool2d(h, 8, 8)
        h = h.view(-1, 196)
        return h


class Generator(nn.Module):
    """Generator."""

    def __init__(self):
        """Generator Builder."""
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(100, 196 * 4 * 4)

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
            nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """Forward pass of Genrator."""
        out = self.fc1(x)
        out = out.view(-1, 196, 4, 4)
        out = self.tconv(out)

        return out
