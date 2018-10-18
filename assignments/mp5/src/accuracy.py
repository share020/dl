"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import torch
import numpy as np

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from numpy import linalg as LA
from torch.autograd import Variable



def calculate_accuracy(net, trainloader, testloader, is_gpu):
    """
    Calculate accuracy for TripletNet model.

        1. Form 2d array: Number of training images * size of embedding
        2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
        3. Perform subtraction between the two 2D arrays
        4, Take L2 norm of the 2d array (after subtraction)
        5. Get the 30 min values (argmin might do the trick)
        6. Repeat for the rest of the embeddings in the test set
    """
    print('==> Retrieve model parameters ...')
    checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    # args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])

    # dictionary of test images with class
    # val_1788.JPEG ==> n04532670
    # val_8463.JPEG ==> n02917067
    class_dict = get_classes()

    # list of traning images names, e.g., "../tiny-imagenet-200/train/n01629819/images/n01629819_238.JPEG"
    training_images = []
    for line in open(triplets_filename):
        line_array = line.split(",")
        training_images.append(line_array[0])

    # get embedded features of training
    embedded_features = []
    for data1, data2, data3 in trainloader:

        if is_gpu:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        # wrap in torch.autograd.Variable
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        embedded_a, _, _ = net(data1, data2, data3)
        embedded_a_numpy = embedded_a.data.cpu().numpy()
        embedded_features.append(embedded_a_numpy)

    # TODO: 1. Form 2d array: Number of training images * size of embedding
    embedded_features_train = np.concatenate(embedded_features, axis=0)

    # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
    for test_id, test_data in enumerate(testloader):

        if is_gpu:
            test_data = test_data.cuda()
        test_data = Variable(test_data)

        embedded_test, _, _ = net(test_data, test_data, test_data)
        embedded_test_numpy = embedded_test_numpy.data.cpu().numpy()
        embedded_features_test = np.tile(embedded_test_numpy, (embedded_features.shape[0], 1))

        # TODO: 3. Perform subtraction between the two 2D arrays
        embedding_diff = embedded_features_train - embedded_features_test

        # TODO: 4, Take L2 norm of the 2d array (after subtraction)
        embedding_norm = LA.norm(embedding_diff, axis=0)

        # TODO: 5. Get the 30 min values (argmin might do the trick)
        min_index = embedding_norm.argsort()[:30]

        # TODO: 6. Repeat for the rest of the embeddings in the test set
        accuracies = []

        # for each image results in min distance
        for idx in min_index:
            correct = 0

            # get result image class
            top_result_image_name = training_images[idx]
            top_result_image_name_class = top_result_image_name.split("/")[3]

            # get test image class
            test_image_name = "val_" + str(test_id) + ".JPEG"
            test_image_class = class_dict[test_image_name]

            if test_image_class == top_result_image_name_class:
                correct += 1

        acc = correct / 30
        accuracies.append(acc)

    avg_acc = sum(accuracies) / len(accuracies)
    print("Test accuracy {}%".format(avg_acc))


def get_classes(filename="../tiny-imagenet-200/val/val_annotations.txt"):
    """
    Get corresponding class name for each val image.


    Args:
        filename: txt file which contains image name and corresponding class name

    Returns:
        class_dict: A dictionary which maps from image name to class name
    """
    class_dict = {}
    for line in open(filename):
        line_array = line.rstrip("\n").split("\t")
        class_dict[line_array[0]] = line_array[1]

    return class_dict







#
