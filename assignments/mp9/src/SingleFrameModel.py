"""
HW9: Human Action Recognition.

Part 1 - Single Frame Model

@author: Zhenye Na
@credit: Logan Courtney
"""


import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2
import argparse


from multiprocessing import Pool

parser = argparse.ArgumentParser()

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--IMAGE_SIZE', type=int, default=224, help='beta1')
parser.add_argument('--NUM_CLASSES', type=int, default=101, help='beta2')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--num_of_epochs', type=int, default=10,
                    help='number of epochs to train')

# parse the arguments
args = parser.parse_args()

IMAGE_SIZE = args.IMAGE_SIZE
NUM_CLASSES = args.NUM_CLASSES
batch_size = args.batch_size
lr = args.lr
num_of_epochs = args.num_of_epochs


data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory=data_directory)


model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, NUM_CLASSES)

for param in model.parameters():
    param.requires_grad_(False)

# for param in model.conv1.parameters():
#     param.requires_grad_(True)
# for param in model.bn1.parameters():
#     param.requires_grad_(True)
# for param in model.layer1.parameters():
#     param.requires_grad_(True)
# for param in model.layer2.parameters():
#     param.requires_grad_(True)
# for param in model.layer3.parameters():
#     param.requires_grad_(True)
for param in model.layer4[2].parameters():
    param.requires_grad_(True)
for param in model.fc.parameters():
    param.requires_grad_(True)

params = []
# for param in model.conv1.parameters():
#     params.append(param)
# for param in model.bn1.parameters():
#     params.append(param)
# for param in model.layer1.parameters():
#     params.append(param)
# for param in model.layer2.parameters():
#     params.append(param)
# for param in model.layer3.parameters():
#     params.append(param)
for param in model.layer4[2].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)

model.cuda()

optimizer = optim.Adam(params, lr=lr)
criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8, maxtasksperchild=200)

for epoch in range(0, num_of_epochs):
    # ---------------------------------------------------------------------- #
    # TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()
    for i in range(0, len(train[0]) - batch_size, batch_size):

        augment = True
        video_list = [(train[0][k], augment)
                      for k in random_indices[i:(batch_size + i)]]
        data = pool_threads.map(loadFrame, video_list)

        next_batch = 0
        for video in data:
            if video.size == 0:  # there was an exception, skip this
                next_batch = 1
        if next_batch == 1:
            continue

        x = np.asarray(data, dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()

        y = train[1][random_indices[i:(batch_size + i)]]
        y = torch.from_numpy(y).cuda()

        output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(y.data).sum()) /
                    float(batch_size)) * 100.0
        if(epoch == 0):
            print(i, accuracy)
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch, time.time() - start_time)

torch.save(model, 'single_frame.model')
pool_threads.close()
pool_threads.terminate()

augment = True
video_list = [(train[0][k], augment) for k in random_indices[i:(batch_size + i)]]
data = pool_threads.map(loadFrame, video_list)

next_batch = 0
for video in data:
    # there was an exception, skip this
    if video.size == 0:
        next_batch = 1
    # if next_batch == 1:
    #     continue

x = np.asarray(data, dtype=np.float32)
x = Variable(torch.FloatTensor(x)).cuda().contiguous()

# TEST
model.eval()
test_accu = []
random_indices = np.random.permutation(len(test[0]))
t1 = time.time()
for i in range(0, len(test[0]) - batch_size, batch_size):
    augment = False
    video_list = [(test[0][k], augment)
                  for k in random_indices[i:(batch_size + i)]]
    data = pool_threads.map(loadFrame, video_list)

    next_batch = 0
    for video in data:
        # there was an exception, skip this batch
        if video.size == 0:
            next_batch = 1
    if next_batch == 1:
        continue

    x = np.asarray(data, dtype=np.float32)
    x = Variable(torch.FloatTensor(x)).cuda().contiguous()

    y = test[1][random_indices[i:(batch_size + i)]]
    y = torch.from_numpy(y).cuda()

    output = model(x)

    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(y.data).sum()) / float(batch_size)) * 100.0
    test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
print('Testing', accuracy_test, time.time() - t1)

# -------------------------------------------------------------------------- #
# Testing

model = torch.load('single_frame.model')
model.cuda()

# save predictions directory
prediction_directory = 'UCF-101-predictions/single_frame/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory + label + '/'):
        os.makedirs(prediction_directory + label + '/')


acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406], np.float32)
std = np.asarray([0.229, 0.224, 0.225], np.float32)
model.eval()


for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi', '.hdf5')
    filename = filename.replace('UCF-101', 'UCF-101-hdf5')

    h = h5py.File(filename, 'r')
    nFrames = len(h['video'])

    data = np.zeros((nFrames, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frame = (frame - mean) / std
        frame = frame.transpose(2, 0, 1)
        data[j, :, :, :] = frame
    h.close()

    prediction = np.zeros((nFrames, NUM_CLASSES), dtype=np.float32)

    loop_i = list(range(0, nFrames, 200))
    loop_i.append(nFrames)

    for j in range(len(loop_i) - 1):
        data_batch = data[loop_i[j]:loop_i[j + 1]]

        with torch.no_grad():
            x = np.asarray(data_batch, dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            output = model(x)

        prediction[loop_i[j]:loop_i[j + 1]] = output.cpu().numpy()

    # saves the `prediction` array in hdf5 format

    filename = filename.replace(
        data_directory + 'UCF-101-hdf5/', prediction_directory)
    if(not os.path.isfile(filename)):
        with h5py.File(filename, 'w') as h:
            h.create_dataset('predictions', data=prediction)

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j]) / np.sum(np.exp(prediction[j]))

    prediction = np.sum(np.log(prediction), axis=0)
    argsort_pred = np.argsort(-prediction)[0:10]

    label = test[1][index]
    confusion_matrix[label, argsort_pred[0]] += 1
    if(label == argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5] == label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:] == label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)'
          % (i, nFrames, time.time() - t1, acc_top1 / (i + 1), acc_top5 / (i + 1), acc_top10 / (i + 1)))


number_of_examples = np.sum(confusion_matrix, axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i, :] = confusion_matrix[i, :] / \
        np.sum(confusion_matrix[i, :])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i], sorted_results[i], number_of_examples[indices[i]])

np.save('single_frame_confusion_matrix.npy', confusion_matrix)
