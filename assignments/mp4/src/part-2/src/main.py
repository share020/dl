"""
HW4: Implement a deep residual neural network for CIFAR100.

Part-2: Fine-tune a pre-trained ResNet-18

Due October 5 at 5:00 PM.

@author: Zhenye Na
"""

import torch
import torchvision
import torch.optim
import torch.nn as nn

import argparse

from torch.optim import lr_scheduler
from utils import *


parser = argparse.ArgumentParser()

# directory
parser.add_argument('--dataroot', type=str, default="../data", help='path to dataset')
parser.add_argument('--ckptroot', type=str, default="../checkpoint/ckpt.t7", help='path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int, default=256, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int, default=256, help='test set input batch size')

# training settings
parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True, help='whether training using GPU')

# model_urls
parser.add_argument('--model_url', type=str, default="https://download.pytorch.org/models/resnet18-5c106cde.pth", help='model url for pretrained model')

# parse the arguments
args = parser.parse_args()


# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# }


def main():
    """Main pipeline for training ResNet model on CIFAR100 Dataset."""
    start_epoch = 0

    # resume training from the last time
    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint ...')
        assert os.path.isdir('../checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckptroot)
        net = checkpoint['net']
        start_epoch = checkpoint['epoch']
    else:
        # start over
        print('==> Load pre-trained ResNet model ...')
        net = resnet18(args.model_url)


    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if args.is_gpu:
        net = net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Loss function, optimizer for fine-tune-able params
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # data loader for CIFAR100
    trainloader, testloader = data_loader(args.dataroot, args.batch_size_train, args.batch_size_test)

    # train pre-trained model on CIFAR100
    model = train_model(net, optimizer, scheduler, criterion, trainloader, testloader, start_epoch, args.epochs, args.is_gpu)



if __name__ == '__main__':
    main()
