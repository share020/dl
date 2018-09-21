"""
HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset.

The convolution network should use
    (A) dropout
    (B) trained with RMSprop or ADAM, and
    (C) data augmentation. 

@author: Zhenye Na
"""

import numpy as np
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse

from cnn import *
from utils import *

parser = argparse.ArgumentParser()

# data root
parser.add_argument('--dataroot', type=str, default="../data", help='path to dataset')
parser.add_argument('--ckptroot', type=str, default="../checkpoint/ckpt.t7", help='path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int, default=128, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int, default=32, help='test set input batch size')


# training settings
parser.add_argument('--resume', type=bool, default=False, help='whether training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True, help='whether training using GPU')

# parse the arguments
opt = parser.parse_args()


# set the seeds
np.random.seed(233)
torch.cuda.manual_seed_all(233)
torch.manual_seed(233)


# Data augmentation

print("==> Data Augmentation ...")

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


# Loading CIFAR10

print("==> Downloading CIFAR10 dataset ...")

trainset    = torchvision.datasets.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size_train, shuffle=True, num_workers=4)

testset     = torchvision.datasets.CIFAR10(root=opt.dataroot, train=False, download=True, transform=transform_test)
testloader  = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size_test, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("==> Initialize CNN model ...")

# resume training from the last time
if opt.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint ...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(opt.ckptroot)
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    # re-start training
    print('==> Building new CNN model ...')
    # Create an instance of the nn.module class defined above:
    net = CNN()
    start_epoch = 0

# For training on GPU, we need to transfer net and data onto the GPU
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
if opt.is_gpu:
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)



for epoch in range(start_epoch, opt.epochs + start_epoch):

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
        calculate_val_accuracy(testloader, opt.is_gpu)
    print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

    if epoch % 50 == 0:
        print('==>  Saving model..')
        state = {
            'net': net.module if opt.is_gpu else net,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '../checkpoint/ckpt.t7')
