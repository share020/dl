"""
HW4: Implement a deep residual neural network for CIFAR100.

Part-1: Build the Residual Network

Due October 10 at 5:00 PM.

@author: Zhenye Na
"""

import os
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from torch.autograd import Variable


def data_loader(dataroot, batch_size_train, batch_size_test):
    """
    Data Loader for CIFAR100 Dataset.

    Args:
        dataroot: data root directory
        batch_size_train: mini-Batch size of training set
        batch_size_test: mini-Batch size of test set

    Returns:
        trainloader: training set loader
        testloader: test set loader
    """
    # Data Augmentation
    print("==> Data Augmentation ...")

    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Loading CIFAR100
    print("==> Preparing CIFAR100 dataset ...")

    trainset = torchvision.datasets.CIFAR100(root=dataroot,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root=dataroot,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=4)

    return trainloader, testloader


def calculate_accuracy(net, loader, is_gpu):
    """
    Calculate accuracy.

    Args:
        loader (torch.utils.data.DataLoader): training / test set loader
        is_gpu (bool): whether to run on GPU

    Returns:
        tuple: overall accuracy
    """
    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


def train(net, criterion, optimizer, trainloader,
          testloader, start_epoch, epochs, is_gpu):
    """
    Training process.

    Args:
        net: ResNet model
        criterion: CrossEntropyLoss
        optimizer: SGD with momentum optimizer
        trainloader: training set loader
        testloader: test set loader
        start_epoch: checkpoint saved epoch
        epochs: training epochs
        is_gpu: whether use GPU

    """
    print("==> Start training ...")

    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            if is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy = calculate_accuracy(net, trainloader, is_gpu)
        test_accuracy = calculate_accuracy(net, testloader, is_gpu)

        print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
            epoch+1, running_loss, train_accuracy, test_accuracy))

        # save model
        if epoch % 50 == 0:
            print('==> Saving model ...')
            state = {
                'net': net.module if is_gpu else net,
                'epoch': epoch,
            }
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')
            torch.save(state, '../checkpoint/ckpt.t7')

    print('==> Finished Training ...')
