"""
HW6: Understanding CNNs and Generative Adversarial Networks.

@author: Zhenye Na
"""


def train(optimizer, start_epoch, epochs):
    """Training process."""

    for epoch in range(start_epoch, start_epoch + epochs):

        if(epoch == 50):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 10.0
        if(epoch == 75):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 100.0
