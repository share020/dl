Fine tune ResNet in Pytorch

- switch the `requires_grad` flags in the frozen base, and no intermediate buffers will be saved
- Constraints from pre-trained models. Note that if you wish to use a pre-trained network, you may be slightly constrained in terms of the architecture you can use for your new dataset.
- Learning rates. It’s common to use a smaller learning rate for ConvNet weights that are being fine-tuned