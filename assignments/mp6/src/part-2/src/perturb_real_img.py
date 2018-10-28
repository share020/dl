"""
HW6: Understanding CNNs and Generative Adversarial Networks.

Part 2: Visualization

@author: Zhenye Na
"""

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from torch.autograd import Variable
from plot import plot


def perturb_real_images(testloader, modelroot, batch_size, cuda):
    """Perturb Real Images."""
    testloader = enumerate(testloader)

    model_pth = os.path.join(modelroot, "cifar10.model")
    model = torch.load(model_pth)
    if cuda:
        model.cuda()
    model.eval()

    batch_idx, (X_batch, Y_batch) = testloader.__next__()
    X_batch = Variable(X_batch, requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1) % 10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()

    # ----------------------------------------------------------------------- #
    # save real images
    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    if not os.path.isdir('../visualization'):
        os.mkdir('../visualization')
    plt.savefig('../visualization/real_images.png', bbox_inches='tight')
    plt.close(fig)

    _, output = model(X_batch)
    # first column has actual prob.
    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(Y_batch.data).sum()) /
                float(batch_size)) * 100.0
    print("Original Image | Accuracy: {}%".format(accuracy))

    # ----------------------------------------------------------------------- #
    # slightly jitter all input images
    criterion = nn.CrossEntropyLoss(reduce=False)
    loss = criterion(output, Y_batch_alternate)

    gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                                    grad_outputs=torch.ones(
                                        loss.size()).cuda(),
                                    create_graph=True, retain_graph=False, only_inputs=True)[0]

    # save gradient jitter
    gradient_image = gradients.data.cpu().numpy()
    gradient_image = (gradient_image - np.min(gradient_image)) / \
        (np.max(gradient_image) - np.min(gradient_image))
    gradient_image = gradient_image.transpose(0, 2, 3, 1)
    fig = plot(gradient_image[0:100])
    plt.savefig('../visualization/gradient_image.png', bbox_inches='tight')
    plt.close(fig)

    # ----------------------------------------------------------------------- #
    # jitter input image
    gradients[gradients > 0.0] = 1.0
    gradients[gradients < 0.0] = -1.0

    gain = 8.0
    X_batch_modified = X_batch - gain * 0.007843137 * gradients
    X_batch_modified[X_batch_modified > 1.0] = 1.0
    X_batch_modified[X_batch_modified < -1.0] = -1.0

    # evaluate new fake images
    _, output = model(X_batch_modified)
    # first column has actual prob
    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(Y_batch.data).sum()) /
                float(batch_size)) * 100.0
    print("Jitter Input Image | Accuracy: {}%".format(accuracy))

    # save fake images
    samples = X_batch_modified.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    plt.savefig('../visualization/jittered_images.png', bbox_inches='tight')
    plt.close(fig)
