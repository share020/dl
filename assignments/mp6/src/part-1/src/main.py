"""
HW6: Understanding CNNs and Generative Adversarial Networks.

@author: Zhenye Na
"""

import os
import torch
import argparse
import torchvision
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from train import Trainer_D, Trainer_GD
from utils import cifar10_loader
from model import Discriminator, Generator


def parse_args():
    parser = argparse.ArgumentParser()

    # trainig command
    parser.add_argument('--option', type=str, default="option1", help='training discriminator with or without generator')

    # directory
    parser.add_argument('--dataroot', type=str, default="../../../data", help='path to dataset')
    parser.add_argument('--ckptroot', type=str, default="../checkpoint/ckpt.t7", help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0., help='beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')

    parser.add_argument('--epochs1', type=int, default=120, help='number of epochs to train without generator')
    parser.add_argument('--epochs2', type=int, default=200, help='number of epochs to train with generator')

    parser.add_argument('--start_epoch', type=int, default=0, help='pre-trained epochs')
    parser.add_argument('--batch_size_train', type=int, default=128, help='training set input batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, help='test set input batch size')

    # parameters for training discriminator and generator gen_train
    parser.add_argument('--n_z', type=int, default=100, help='number of hidden units')
    parser.add_argument('--gen_train', type=int, default=5, help='number of epochs that trains generator while training discriminator')

    # training settings
    parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
    parser.add_argument('--cuda', type=bool, default=False, help='whether training using cudatoolkit')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipleline that implements Generative Adversarial Networks in Pytorch"""
    args = parse_args()

    # load cifar10 dataset
    trainloader, testloader = cifar10_loader(args.dataroot, args.batch_size_train, args.batch_size_test)

    # Train the Discriminator without the Generator
    if args.option == "option1":
        print("Train the Discriminator without the Generator ...")
        model = Discriminator()
        if args.cuda:
            model = nn.DataParallel(model).cuda()
            cudnn.benchmark=True
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        # train
        trainer_d = Trainer_D(model, criterion, optimizer, trainloader, testloader,
                              args.start_epoch, args.epochs1, args.cuda, args.batch_size_train, args.lr)
        trainer_d.train()

    # Train the Discriminator with the Generator
    else:
        print("Train the Discriminator with the Generator ...")
        aD, aG = Discriminator(), Generator()
        if args.cuda:
            aD, aG = nn.DataParallel(aD).cuda(), nn.DataParallel(aG).cuda()
            cudnn.benchmark=True

        optimizer_g = torch.optim.Adam(aG.parameters(),
                                       lr=args.lr,
                                       betas=(args.beta1, args.beta2))
        optimizer_d = torch.optim.Adam(aD.parameters(),
                                       lr=args.lr,
                                       betas=(args.beta1, args.beta2))

        criterion = nn.CrossEntropyLoss()

        # train
        trainer_gd = Trainer_GD(aD, aG, criterion, optimizer_d, optimizer_g, trainloader, testloader, args.batch_size_train, args.gen_train, args.cuda, args.n_z, args.start_epoch, args.epochs2)
        trainer_gd.train()



    # # resume training from the last time
    # if args.resume:
    #     # Load checkpoint
    #     print('==> Resuming training from checkpoint ...')
    #     checkpoint = torch.load(args.ckptroot)
    #     args.start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     net.load_state_dict(checkpoint['state_dict'])
    #     print("==> Loaded checkpoint '{}' (epoch {})".format(args.ckptroot, checkpoint['epoch']))
    #
    # else:
    #     # start over
    #     print('==> Building new TripletNet model ...')
    #     net = TripletNet(resnet101())






if __name__ == '__main__':
    main()
