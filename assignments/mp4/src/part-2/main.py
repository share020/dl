"""
HW4: Implement a deep residual neural network for CIFAR100.

Part-2: Fine-tune a pre-trained ResNet-18

Due October 5 at 5:00 PM.

@author: Zhenye Na
"""

import torch
import torchvision


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

def resnet18(pretrained = True) :
    
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])
    
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18'], model_dir = './'))
    return model




