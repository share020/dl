"""
HW6: Understanding CNNs and Generative Adversarial Networks.
Part 2: Visualization

@author: Zhenye Na
"""

import argparse


from utils import cifar10_loader
from perturb_real_img import perturb_real_images
from syn_img import syn_img
from syn_features import syn_features

def parse_args():
    parser = argparse.ArgumentParser()

    # trainig command
    parser.add_argument('--option', type=str, default="option1", help='Perturb Real Images')

    # directory
    parser.add_argument('--dataroot', type=str, default="../../../data", help='path to dataset')
    parser.add_argument('--modelroot', type=str, default="../../model", help='path to model of discriminator without generator')
    parser.add_argument('--ckptroot', type=str, default="../model/", help='path to checkpoint')

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
    parser.add_argument('--cuda', type=bool, default=True, help='whether training using cudatoolkit')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline for GAN Visualization."""
    args = parse_args()

    testloader = cifar10_loader(args.dataroot, args.batch_size_test)

    if args.option == "option1":
        # Perturb Real Images
        perturb_real_images(testloader, args.modelroot, args.batch_size_test, args.cuda)
        print("==> Perturb Real Images done ...")
    elif args.option == "option2":
        # Synthetic Images Maximizing Classification Output
        syn_img(testloader, args.modelroot, args.cuda)
        print("==> Synthetic Images Maximizing Classification Output done ...")
    elif args.option == "option3":
        # Synthetic Features Maximizing Features at Various Layers
        syn_features(testloader, args.modelroot, args.batch_size_test, args.cuda)
        print("==> Synthetic Features Maximizing Features at Various Layers done ...")
    else:
        print("==> No such option ...")


if __name__ == '__main__':
    main()
