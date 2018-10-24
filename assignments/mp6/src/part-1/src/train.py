"""
HW6: Understanding CNNs and Generative Adversarial Networks.
Part-1: Training a GAN on CIFAR10

@author: Zhenye Na
"""

import os
import time
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.autograd import Variable
from utils import calc_gradient_penalty, calculate_accuracy
from plot import plot

class Trainer_D(object):
    """docstring for Trainer."""
    def __init__(self, model, criterion, optimizer, trainloader, testloader, start_epoch, epochs, cuda, batch_size, learning_rate):
        super(Trainer_D, self).__init__()

        self.cuda          = cuda
        self.model         = model
        self.epochs        = epochs
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.batch_size    = batch_size
        self.testloader    = testloader
        self.trainloader   = trainloader
        self.start_epoch   = start_epoch
        self.learning_rate = learning_rate

    def train(self):
        """Training Discriminator without Generator."""
        print("==> Start training ...")
        running_loss = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            # learning rate decay
            if epoch == 40:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate / 10.0
            if epoch == 80:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate / 100.0

            for batch_idx, (X_train_batch, Y_train_batch) in enumerate(self.trainloader):

                if Y_train_batch.shape[0] < self.batch_size:
                    continue

                if self.cuda:
                    X_train_batch = X_train_batch.cuda()
                    Y_train_batch = Y_train_batch.cuda()
                X_train_batch, Y_train_batch = Variable(X_train_batch), Variable(Y_train_batch)
                _, output = self.model(X_train_batch)

                loss = self.criterion(output, Y_train_batch)
                self.optimizer.zero_grad()
                loss.backward()

                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.optimizer.state[p]
                        if('step' in state and state['step']>=1024):
                            state['step'] = 1000
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]

            # Normalizing the loss by the total number of train batches
            running_loss /= len(self.trainloader)

            # Calculate training/test set accuracy of the existing model
            train_accuracy = calculate_accuracy(self.model, self.trainloader, self.cuda)
            test_accuracy = calculate_accuracy(self.model, self.testloader, self.cuda)
            print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch+1, running_loss, train_accuracy, test_accuracy))

            if epoch % 5 == 0:
                print("==> Saving model at epoch: {}".format(epoch))
                if not os.path.isdir('../model'):
                    os.mkdir('../model')
                torch.save(self.model, '../model/cifar10.model')



class Trainer_GD(object):
    """docstring for Trainer_GD."""
    def __init__(self, aD, aG, criterion, optimizer_d, optimizer_g, trainloader, testloader, batch_size, gen_train, cuda, n_z, start_epoch, epochs):
        super(Trainer_GD, self).__init__()

        self.aD          = aD
        self.aG          = aG
        self.n_z         = n_z
        self.cuda        = cuda
        self.epochs      = epochs
        self.gen_train   = gen_train
        self.criterion   = criterion
        self.testloader  = testloader
        self.trainloader = trainloader
        self.start_epoch = start_epoch
        self.batch_size  = batch_size
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g


    def train(self):
        """Training Discriminator with Generator."""
        start_time = time.time()
        n_classes = 10

        # before epoch training loop starts
        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []
        loss5 = []
        acc1  = []

        np.random.seed(352)
        label = np.asarray(list(range(10)) * 10)
        noise = np.random.normal(0, 1, (100, self.n_z))
        label_onehot = np.zeros((100, n_classes))
        label_onehot[np.arange(100), label] = 1
        noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
        noise = noise.astype(np.float32)

        save_noise = torch.from_numpy(noise)
        if self.cuda:
            save_noise = save_noise.cuda()
        save_noise = Variable(save_noise)

        # Train the model
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            # turn models to `train` mode
            self.aG.train()
            self.aD.train()

            for batch_idx, (X_train_batch, Y_train_batch) in enumerate(self.trainloader):

                if Y_train_batch.shape[0] < self.batch_size:
                    continue

                # train G
                if batch_idx % self.gen_train == 0:
                    for p in self.aD.parameters():
                        p.requires_grad_(False)

                    self.aG.zero_grad()

                    label = np.random.randint(0, n_classes, self.batch_size)
                    noise = np.random.normal(0, 1, (self.batch_size, self.n_z))
                    label_onehot = np.zeros((self.batch_size, n_classes))
                    label_onehot[np.arange(self.batch_size), label] = 1
                    noise[np.arange(self.batch_size), :n_classes] = label_onehot[np.arange(self.batch_size)]
                    noise = noise.astype(np.float32)
                    noise = torch.from_numpy(noise)
                    if self.cuda:
                        noise = noise.cuda()
                    noise = Variable(noise)
                    # noise = Variable(noise).cuda()
                    if self.cuda:
                        fake_label = Variable(torch.from_numpy(label)).cuda()
                    else:
                        fake_label = Variable(torch.from_numpy(label))
                    # fake_label = Variable(torch.from_numpy(label)).cuda()

                    fake_data = self.aG(noise)
                    gen_source, gen_class = self.aD(fake_data)

                    gen_source = gen_source.mean()
                    gen_class = self.criterion(gen_class, fake_label)

                    gen_cost = -gen_source + gen_class
                    gen_cost.backward()

                    for group in self.optimizer_g.param_groups:
                        for p in group['params']:
                            state = self.optimizer_g.state[p]
                            if('step' in state and state['step']>=1024):
                                state['step'] = 1000
                    self.optimizer_g.step()

                # train D
                for p in self.aD.parameters():
                    p.requires_grad_(True)
                self.aD.zero_grad()

                # train discriminator with input from generator
                label = np.random.randint(0, n_classes, self.batch_size)
                noise = np.random.normal(0, 1, (self.batch_size, self.n_z))
                label_onehot = np.zeros((self.batch_size, n_classes))
                label_onehot[np.arange(self.batch_size), label] = 1
                noise[np.arange(self.batch_size), :n_classes] = label_onehot[np.arange(self.batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise)
                if self.cuda:
                    noise = noise.cuda()
                noise = Variable(noise)

                if self.cuda:
                    fake_label = Variable(torch.from_numpy(label)).cuda()
                else:
                    fake_label = Variable(torch.from_numpy(label))

                with torch.no_grad():
                    fake_data = self.aG(noise)

                disc_fake_source, disc_fake_class = self.aD(fake_data)

                disc_fake_source = disc_fake_source.mean()
                disc_fake_class = self.criterion(disc_fake_class, fake_label)

                # train discriminator with input from the discriminator
                if self.cuda:
                    real_data, real_label = X_train_batch.cuda(), Y_train_batch.cuda()
                else:
                    real_data, real_label = X_train_batch, Y_train_batch
                real_data, real_label = Variable(real_data), Variable(real_label)
                # real_data = Variable(X_train_batch).cuda()
                # real_label = Variable(Y_train_batch).cuda()

                disc_real_source, disc_real_class = self.aD(real_data)

                prediction = disc_real_class.data.max(1)[1]
                accuracy = (float(prediction.eq(real_label.data).sum()) / float(self.batch_size)) * 100.0

                disc_real_source = disc_real_source.mean()
                disc_real_class = self.criterion(disc_real_class, real_label)

                gradient_penalty = calc_gradient_penalty(self.aD, real_data, fake_data, self.batch_size, self.cuda)

                disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
                disc_cost.backward()

                for group in self.optimizer_d.param_groups:
                    for p in group['params']:
                        state = self.optimizer_d.state[p]
                        if('step' in state and state['step']>=1024):
                            state['step'] = 1000
                self.optimizer_d.step()

                # within the training loop
                loss1.append(gradient_penalty.item())
                loss2.append(disc_fake_source.item())
                loss3.append(disc_real_source.item())
                loss4.append(disc_real_class.item())
                loss5.append(disc_fake_class.item())
                acc1.append(accuracy)
                if batch_idx % 50 == 0:
                    print("Trainig epoch: {} | Accuracy: {} | Batch: {} | Gradient penalty: {} | Discriminator fake source: {} | Discriminator real source: {} | Discriminator real class: {} | Discriminator fake class: {}".format(epoch, np.mean(acc1), batch_idx, np.mean(loss1), np.mean(loss2), np.mean(loss3), np.mean(loss4), np.mean(loss5)))

            # Test the model
            self.aD.eval()
            with torch.no_grad():
                test_accu = []
                for batch_idx, (X_test_batch, Y_test_batch) in enumerate(self.testloader):
                    if self.cuda:
                        X_test_batch, Y_test_batch = X_test_batch.cuda(), Y_test_batch.cuda()
                    X_test_batch, Y_test_batch = Variable(X_test_batch), Variable(Y_test_batch)

                    with torch.no_grad():
                        _, output = self.aD(X_test_batch)

                    # first column has actual prob.
                    prediction = output.data.max(1)[1]
                    accuracy = (float(prediction.eq(Y_test_batch.data).sum()) / float(self.batch_size)) * 100.0
                    test_accu.append(accuracy)
                    accuracy_test = np.mean(test_accu)
            # print('Testing', accuracy_test, time.time() - start_time)
            print("Testing accuracy: {} | Eplased time: {}".format(accuracy_test, time.time() - start_time))

            # save output
            with torch.no_grad():
                self.aG.eval()
                samples = self.aG(save_noise)
                samples = samples.data.cpu().numpy()
                samples += 1.0
                samples /= 2.0
                samples = samples.transpose(0,2,3,1)
                self.aG.train()

            fig = plot(samples)
            if not os.path.isdir('../output'):
                os.mkdir('../output')
            plt.savefig('../output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
            plt.close(fig)

            if (epoch + 1) % 1 == 0:
                torch.save(self.aG, '../model/tempG.model')
                torch.save(self.aD, '../model/tempD.model')
