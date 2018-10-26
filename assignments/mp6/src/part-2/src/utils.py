"""
HW6: Understanding CNNs and Generative Adversarial Networks.

Part 2: Visualization

@author: Zhenye Na
"""

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms


def cifar10_loader(root, batch_size):
    """
    CIFAR10 dataset loader.

    Args:
        root: data root directory
        batch_size: batch size to load testset
    """
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return testloader
