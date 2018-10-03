"""
HW4: Implement a deep residual neural network for CIFAR100.

Part-2: Fine-tune a pre-trained ResNet-18

Due October 5 at 5:00 PM.

@author: Zhenye Na
"""

import torch
import torch.nn as nn

class FineTune(nn.Module):

    def __init__(self, resnet, num_classes):
        super(FineTune, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )

        # Freeze those weights
        for param in self.features.parameters():
            param.requires_grad = False


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
