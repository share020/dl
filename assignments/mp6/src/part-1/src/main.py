"""
HW6: Understanding CNNs and Generative Adversarial Networks.

@author: Zhenye Na
"""

import os
import torch
import torchvision
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn



import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--dataroot', type=str, default="../data", help='path to dataset')
    parser.add_argument('--ckptroot', type=str, default="../checkpoint/ckpt.t7", help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, help='pre-trained epochs')
    parser.add_argument('--batch_size_train', type=int, default=256, help='training set input batch size')
    parser.add_argument('--batch_size_test', type=int, default=256, help='test set input batch size')

    # training settings
    parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
    parser.add_argument('--is_gpu', type=bool, default=True, help='whether training using GPU')

    # parse the arguments
    args = parser.parse_args()

    return args



def main():
    """Main pipleline that implements Generative Adversarial Networks in Pytorch"""
    args = parse_args()



    model = discriminator()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)











    # customized ----------

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if args.is_gpu:
        print("==> Initialize CUDA support for TripletNet model ...")
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # resume training from the last time
    if args.resume:
        # Load checkpoint
        print('==> Resuming training from checkpoint ...')
        checkpoint = torch.load(args.ckptroot)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        print("==> Loaded checkpoint '{}' (epoch {})".format(args.ckptroot, checkpoint['epoch']))

    else:
        # start over
        print('==> Building new TripletNet model ...')
        net = TripletNet(resnet101())






if __name__ == '__main__':
    main()
