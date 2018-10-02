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

def resnet18(model_urls, pretrained=True) :
    """Load pre-trained ResNet-18 model in Pytorch."""
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])

    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls, model_dir = './'))
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 100)
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

    normalize = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # # Normalize training set together with augmentation
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    # ])
    #
    # # Normalize test set same as training set without augmentation
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    # ])

    # Loading CIFAR100
    print("==> Preparing CIFAR100 dataset ...")

    trainset = torchvision.datasets.CIFAR100(root=dataroot,
                                             train=True,
                                             download=True,
                                             transform=normalize)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=dataroot,
                                            train=False,
                                            download=True,
                                            transform=normalize)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return trainloader, testloader


def train_model(net, optimizer, criterion, trainloader, testloader, start_epoch, epochs, is_gpu):
    """
    Training.

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

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if epoch > 16:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000
            optimizer.step()


        # save model
        if epoch % 50 == 0:
            print('==> Saving model ...')
            state = {
                'net': net.module if opt.is_gpu else net,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, '../checkpoint/ckpt.t7')

    print('==> Finished Training ...')
