"""Tutorial for Distributed Training using Pytorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.distributed as dist


import os
import subprocess
from mpi4py import MPI

import time
import h5py


# from torchvision import datasets, transforms
# from torchvision import transforms
# from PIL import Image

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
    ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor


# load CIFAR10 data
TinyImageNet_data = h5py.File('TinyImageNet2.hdf5', 'r')
x_train = np.float32(TinyImageNet_data['X_train'][:])
y_train = np.int32(np.array(TinyImageNet_data['Y_train'][:]))
x_test = np.float32(TinyImageNet_data['X_test'][:])
y_test = np.int32(np.array(TinyImageNet_data['Y_test'][:]))
TinyImageNet_data.close()
L_Y_train = len(y_train)
L_Y_test = len(y_test)


class Net2(nn.Module):
    """Simple net for distributed training using Pytorch."""

    def __init__(self):
        """Simple net Builder."""
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, 4, padding=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, 4, padding=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, 4, padding=(2, 2))
        self.conv5 = nn.Conv2d(128, 128, 4, padding=(2, 2))
        self.conv6 = nn.Conv2d(128, 128, 4)
        self.conv7 = nn.Conv2d(128, 128, 4)
        self.conv8 = nn.Conv2d(128, 128, 4)

        # self.conv3 = nn.Conv2d(64, 64, 4, padding=(2, 2))
        # self.conv4 = nn.Conv2d(64, 64, 4, padding=(2, 2))
        # self.conv5 = nn.Conv2d(64, 64, 4, padding=(2, 2))
        # self.conv6 = nn.Conv2d(64, 64, 4)
        # self.conv7 = nn.Conv2d(64, 64, 4)
        # self.conv8 = nn.Conv2d(64, 64, 4)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.1)
        self.dropout3 = torch.nn.Dropout(p=0.1)
        self.dropout4 = torch.nn.Dropout(p=0.1)
        self.dropout5 = torch.nn.Dropout(p=0.1)
        self.dropout6 = torch.nn.Dropout(p=0.1)
        self.dropout7 = torch.nn.Dropout(p=0.1)
        self.dropout8 = torch.nn.Dropout(p=0.1)

        self.Bnorm1 = torch.nn.BatchNorm2d(64)
        self.Bnorm2 = torch.nn.BatchNorm2d(128)
        self.Bnorm3 = torch.nn.BatchNorm2d(128)
        self.Bnorm4 = torch.nn.BatchNorm2d(128)
        self.Bnorm5 = torch.nn.BatchNorm2d(128)

        # self.fc1 = nn.Linear(9 * 9 * 128, 1000)
        # self.fc2 = nn.Linear(1000, 1000)
        # self.fc3 = nn.Linear(1000, 200)
        self.fc1 = nn.Linear(9 * 9 * 128, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 200)

        # self.VerticalFlip = transforms.RandomVerticalFlip(p=0.1)
        # self.HorizontalFlip = transforms.RandomHorizontalFlip(p=0.1)
        # Rotate0 = transforms.RandomRotation(30)
        # self.RotateRandom = transforms.RandomApply(Rotate0, p=0.1)

    def forward(self, x):
        """Forward pass."""
        # x = self.VerticalFlip(x)
        # x = self.HorizontalFlip(x)

        x = self.Bnorm1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        p = self.dropout1(self.pool(x))

        x = self.Bnorm2(F.relu(self.conv3(p)))
        x = F.relu(self.conv4(x))
        p = self.dropout2(self.pool(x))

        x = self.Bnorm3(F.relu(self.conv5(p)))
        x = self.dropout3(F.relu(self.conv6(x)))
        x = self.Bnorm4(F.relu(self.conv7(x)))
        x = self.Bnorm5(F.relu(self.conv8(x)))

        x = self.dropout6(x)

        x = x.view(-1, 9 * 9 * 128)

        x = self.dropout7(F.relu(self.fc1(x)))
        x = self.dropout8(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


model = Net2()

# Make sure that all nodes have the same model
for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0 / np.sqrt(np.float(num_nodes))


model.cuda()

Path_Save = '/projects/sciteam/bahp/RNN/TinyImageNetModel'
# torch.save(model.state_dict(), Path_Save)
# model.load_state_dict(torch.load(Path_Save))


LR = 0.001
batch_size = 100
Num_Epochs = 1000


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)


I_permutation = np.random.permutation(L_Y_train)
x_train = x_train[I_permutation, :]
y_train = y_train[I_permutation]


L_Y_test = round(0.1 * L_Y_train)
x_test = x_train[0:L_Y_test, :]
y_test = y_train[0:L_Y_test]

x_train = x_train[L_Y_test + 1:, :]
y_train = y_train[L_Y_test + 1:]
L_Y_train = len(y_train)


for epoch in range(Num_Epochs):
    time1 = time.time()
    time000 = time.time()
    I_permutation = np.random.permutation(L_Y_train)
    x_train = x_train[I_permutation, :]
    y_train = y_train[I_permutation]
    time111 = time.time()
    time222 = time111 - time000
    # print(time222)
    model.train()
    for i in range(0, L_Y_train, batch_size):
        x_train_batch = torch.FloatTensor(x_train[i:i + batch_size, :])
        y_train_batch = torch.LongTensor(y_train[i:i + batch_size])
        # data, target = Variable(x_train_batch), Variable(y_train_batch)
        data, target = Variable(x_train_batch).cuda(
        ), Variable(y_train_batch).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()    # calc gradients

        for param in model.parameters():
            # print(param.grad.data)
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()

        optimizer.step()  # update gradients

    model.eval()
    # Train Loss
    counter = 0
    train_accuracy_sum = 0.0
    for i in range(0, 10000, batch_size):
        x_train_batch = torch.FloatTensor(x_train[i:i + batch_size, :])
        y_train_batch = torch.LongTensor(y_train[i:i + batch_size])
        data, target = Variable(x_train_batch).cuda(
        ), Variable(y_train_batch).cuda()
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) /
                    float(batch_size)) * 100.0
        counter += 1
        train_accuracy_sum = train_accuracy_sum + accuracy
    train_accuracy_ave = train_accuracy_sum / float(counter)

    # Test Loss
    counter = 0
    test_accuracy_sum = 0.0
    for i in range(0, L_Y_test, batch_size):
        x_test_batch = torch.FloatTensor(x_test[i:i + batch_size, :])
        y_test_batch = torch.LongTensor(y_test[i:i + batch_size])
        data, target = Variable(
            x_test_batch).cuda(), Variable(y_test_batch).cuda()
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) /
                    float(batch_size)) * 100.0
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum / float(counter)
    time2 = time.time()
    time_elapsed = time2 - time1
    print(epoch, test_accuracy_ave, train_accuracy_ave, time_elapsed)
    # save model
    torch.save(model.state_dict(), Path_Save)
