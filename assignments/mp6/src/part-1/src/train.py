"""
HW6: Understanding CNNs and Generative Adversarial Networks.

@author: Zhenye Na
"""

import time
import torch

from torch.autograd import Variable
from utils import calc_gradient_penalty, calculate_accuracy


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
        """Training process."""
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
                torch.save(model, '../model/cifar10.model')



# class Trainer_GD(object):
#     """docstring for Trainer_GD."""
#     def __init__(self, arg):
#         super(Trainer_GD, self).__init__()
#         self.arg = arg
#
#         self.gen_train = gen_train
#
#     def train():
#         start_time = time.time()
#
#         # Train the model
#         for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
#             # turn models to `train` mode
#             aG.train()
#             aD.train()
#
#             for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
#
#                 if Y_train_batch.shape[0] < batch_size:
#                     continue
#
#                 # train G
#                 if batch_idx % self.gen_train == 0:
#                     for p in aD.parameters():
#                         p.requires_grad_(False)
#
#                     aG.zero_grad()
#
#                     label = np.random.randint(0,n_classes,batch_size)
#                     noise = np.random.normal(0,1,(batch_size,n_z))
#                     label_onehot = np.zeros((batch_size,n_classes))
#                     label_onehot[np.arange(batch_size), label] = 1
#                     noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
#                     noise = noise.astype(np.float32)
#                     noise = torch.from_numpy(noise)
#                     noise = Variable(noise).cuda()
#                     fake_label = Variable(torch.from_numpy(label)).cuda()
#
#                     fake_data = aG(noise)
#                     gen_source, gen_class  = aD(fake_data)
#
#                     gen_source = gen_source.mean()
#                     gen_class = criterion(gen_class, fake_label)
#
#                     gen_cost = -gen_source + gen_class
#                     gen_cost.backward()
#
#                     optimizer_g.step()
#
#
#
#                 # train D
#                 for p in aD.parameters():
#                     p.requires_grad_(True)
#
#                 aD.zero_grad()
#
#                 # train discriminator with input from generator
#                 label = np.random.randint(0,n_classes,batch_size)
#                 noise = np.random.normal(0,1,(batch_size,n_z))
#                 label_onehot = np.zeros((batch_size,n_classes))
#                 label_onehot[np.arange(batch_size), label] = 1
#                 noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
#                 noise = noise.astype(np.float32)
#                 noise = torch.from_numpy(noise)
#                 noise = Variable(noise).cuda()
#                 fake_label = Variable(torch.from_numpy(label)).cuda()
#                 with torch.no_grad():
#                     fake_data = aG(noise)
#
#                 disc_fake_source, disc_fake_class = aD(fake_data)
#
#                 disc_fake_source = disc_fake_source.mean()
#                 disc_fake_class = criterion(disc_fake_class, fake_label)
#
#                 # train discriminator with input from the discriminator
#                 real_data = Variable(X_train_batch).cuda()
#                 real_label = Variable(Y_train_batch).cuda()
#
#                 disc_real_source, disc_real_class = aD(real_data)
#
#                 prediction = disc_real_class.data.max(1)[1]
#                 accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0
#
#                 disc_real_source = disc_real_source.mean()
#                 disc_real_class = criterion(disc_real_class, real_label)
#
#                 gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
#
#                 disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
#                 disc_cost.backward()
#
#                 optimizer_d.step()
#
#
#
#
#
#
#
#             # within the training loop
#             loss1.append(gradient_penalty.item())
#             loss2.append(disc_fake_source.item())
#             loss3.append(disc_real_source.item())
#             loss4.append(disc_real_class.item())
#             loss5.append(disc_fake_class.item())
#             acc1.append(accuracy)
#             if((batch_idx%50)==0):
#                 print(epoch, batch_idx, "%.2f" % np.mean(loss1),
#                                         "%.2f" % np.mean(loss2),
#                                         "%.2f" % np.mean(loss3),
#                                         "%.2f" % np.mean(loss4),
#                                         "%.2f" % np.mean(loss5),
#                                         "%.2f" % np.mean(acc1))
#
#
#     def test():
#         # Test the model
#         aD.eval()
#         with torch.no_grad():
#             test_accu = []
#             for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
#                 if self.cuda:
#                     X_test_batch, Y_test_batch = X_test_batch.cuda(), Y_test_batch.cuda()
#                 X_test_batch, Y_test_batch = Variable(X_test_batch), Variable(Y_test_batch)
#
#                 with torch.no_grad():
#                     _, output = aD(X_test_batch)
#
#                 # first column has actual prob.
#                 prediction = output.data.max(1)[1]
#                 accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
#                 test_accu.append(accuracy)
#                 accuracy_test = np.mean(test_accu)
#         print('Testing',accuracy_test, time.time()-start_time)
#
#
#
# ### save output
# with torch.no_grad():
#     aG.eval()
#     samples = aG(save_noise)
#     samples = samples.data.cpu().numpy()
#     samples += 1.0
#     samples /= 2.0
#     samples = samples.transpose(0,2,3,1)
#     aG.train()
#
# fig = plot(samples)
# plt.savefig('../output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
# plt.close(fig)
#
# if(((epoch+1)%1)==0):
#     torch.save(aG,'tempG.model')
#     torch.save(aD,'tempD.model')
