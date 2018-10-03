"""
HW4: Implement a deep residual neural network for CIFAR100.

Part-2: Fine-tune a pre-trained ResNet-18

Due October 5 at 5:00 PM.

@author: Zhenye Na
"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from torch.autograd import Variable
from model import *


def resnet18(model_urls, pretrained=True) :
    """Load pre-trained ResNet-18 model in Pytorch."""
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])

    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls, model_dir = '../'))
        model = FineTune(model, num_classes=100)
    return model


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
        classes: classes names
    """
    # Data Augmentation
    print("==> Data Augmentation ...")

    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Loading CIFAR100
    print("==> Preparing CIFAR100 dataset ...")

    trainset = torchvision.datasets.CIFAR100(root=dataroot,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root=dataroot,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader


def calculate_accuracy(loader, is_gpu):
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


def train_model(net, optimizer, scheduler, criterion, trainloader, testloader, start_epoch, epochs, is_gpu):
    """
    Training process.

    Args:
        net: ResNet model
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        trainloader: training set loader
        testloader: test set loader
        start_epoch: checkpoint saved epoch
        epochs: training epochs
        is_gpu: whether use GPU

    """
    print("==> Start training ...")

    # switch to train mode
    net.train()

    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        scheduler.step()

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # if epoch > 16:
            #     for group in optimizer.param_groups:
            #         for p in group['params']:
            #             state = optimizer.state[p]
            #             if state['step'] >= 1024:
            #                 state['step'] = 1000
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy = calculate_accuracy(trainloader, is_gpu)
        test_accuracy = calculate_accuracy(testloader, is_gpu)

        print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch+1, running_loss, train_accuracy, test_accuracy))


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
