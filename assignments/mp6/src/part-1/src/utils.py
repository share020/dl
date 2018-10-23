"""
HW6: Understanding CNNs and Generative Adversarial Networks.

@author: Zhenye Na
"""

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from torch.autograd import Variable

def cifar10_loader(root, batch_size_train, batch_size_test):
    """CIFAR10 dataset Loader.

    Args:
        root
        batch_size_train
        batch_size_test

    Returns:

    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.ColorJitter(
            brightness=0.1 * torch.randn(1),
            contrast=0.1 * torch.randn(1),
            saturation=0.1 * torch.randn(1),
            hue=0.1 * torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=8)

    return trainloader, testloader


def calculate_accuracy(model, loader, cuda):
    """
    Calculate accuracy.
    Args:
        loader (torch.utils.data.DataLoader): training / test set loader
        cuda (bool): whether to initialize cudatoolkit
    Returns:
        tuple: overall accuracy
    """
    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        if cuda:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, cuda):
    """
    Gradient penalty.

    Args:
        netD: Discriminator
        real_data: real images
        fake_data: generated images
        batch_size: batch size
    """
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    if cuda:
        alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    if cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)
# .cuda()
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty





















#
