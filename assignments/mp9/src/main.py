"""
HW9: Human Action Recognition.

@author: Zhenye Na
@credit: Logan Courtney
"""

import argparse


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()

    # trainig command
    parser.add_argument('--option', type=str,
                        default="option1", help='which part')

    # directory
    parser.add_argument('--dataroot', type=str,
                        default="../../../data", help='path to dataset')
    parser.add_argument('--ckptroot', type=str,
                        default="../model/", help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0., help='beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int,
                        default=0, help='pre-trained epochs')
    parser.add_argument('--batch_size_train', type=int,
                        default=16, help='training set input batch size')
    parser.add_argument('--batch_size_test', type=int,
                        default=16, help='test set input batch size')

    # training settings
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether re-training from ckpt')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether training using cudatoolkit')

    # parse the arguments
    args = parser.parse_args()
    return args


def main():
    """Main pipeline for Action Recognition."""
    args = parse_args()
