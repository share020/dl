"""
HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset.


@author: Zhenye Na
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import os
import argparse

from cnn import *
from utils import *

parser = argparse.ArgumentParser()

# data root
parser.add_argument('--dataroot', type=str, default="../MNISTdata.hdf5", help='path to dataset')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

# training settings
parser.add_argument('--resume', type=bool, default=False, help='whether training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True, help='whether training using GPU')


# parse the arguments
opt = parser.parse_args()

print(">>> Data Augmentation ...")

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print(">>> Downloading CIFAR10 dataset ...")

# get the dataset
trainset    = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

testset    = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=2)

# classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print(">>> Initialize CNN model ...")


if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../checkpoint/ckpt.t7')
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # Create an instance of the nn.module class defined above:
    net = CNN()
    start_epoch = 0

# For training on GPU, we need to transfer net and data onto the GPU
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
if opt.is_gpu:
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# 3. Define a Loss function and optimizer
# Here I use Cross-Entropy loss and Adam algorithm.
# The CrossEntropyLoss criterion already includes softmax within its
# implementation.

criterion = nn.CrossEntropyLoss()

# Tune the learning rate.
# See whether the momentum is useful or not
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)



for epoch in range(start_epoch, EPOCHS + start_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        if opt.is_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)
    print('[%d] loss: %.3f' % (epoch + 1, running_loss))

    # Scale of 0.0 to 100.0
    # Calculate validation set accuracy of the existing model
    val_accuracy, val_classwise_accuracy = \
        calculate_val_accuracy(testloader, IS_GPU)
    print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

    # # Optionally print classwise accuracies
    # for c_i in range(TOTAL_CLASSES):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[c_i], 100 * val_classwise_accuracy[c_i]))

    train_loss_over_epochs.append(running_loss)
    val_accuracy_over_epochs.append(val_accuracy)

    if epoch % 50 == 0:
        print('==>  Saving model..')
        state = {
            'net': net.module if IS_GPU else net,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
