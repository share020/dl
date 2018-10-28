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
    """Parse arguments."""
    parser = argparse.ArgumentParser()

    # trainig command
    parser.add_argument('--option', type=str, default="option3", help='Perturb Real Images / Synthetic Images Maximizing Classification Output / Synthetic Features Maximizing Features at Various Layers')

    # directory
    parser.add_argument('--dataroot', type=str, default="../../../data", help='path to dataset')
    parser.add_argument('--modelroot', type=str, default="../../model", help='path to saved model')

    # hyperparameters settings
    parser.add_argument('--batch_size_test', type=int, default=128, help='test set input batch size')

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
