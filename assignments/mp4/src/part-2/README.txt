Fine tune ResNet in Pytorch

- switch the `requires_grad` flags in the frozen base, and no intermediate buffers will be saved
- Constraints from pre-trained models. Note that if you wish to use a pre-trained network, you may be slightly constrained in terms of the architecture you can use for your new dataset.
- Learning rates. It’s common to use a smaller learning rate for ConvNet weights that are being fine-tuned


----------------------------------------
Begin Torque Prologue on nid25428
at Mon Oct  8 11:09:50 CDT 2018
Job Id:                 9086117.bw
Username:               traXXX
Group:                  TRAIN_bauh
Job name:               cs598-mp4-part2
Requested resources:    neednodes=1:ppn=16:xk,nodes=1:ppn=16:xk,walltime=06:00:00
Queue:                  normal
Account:                bauh
End Torque Prologue:  0.117 elapsed
----------------------------------------




==> Load pre-trained ResNet model ...
==> Data Augmentation ...
==> Preparing CIFAR100 dataset ...
Files already downloaded and verified
Files already downloaded and verified
==> Start training ...
Iteration: 1 | Loss: 3.102943994200138 | Training accuracy: 35.358% | Test accuracy: 42.99%
==> Saving model ...
Iteration: 2 | Loss: 2.40171246303012 | Training accuracy: 42.136% | Test accuracy: 50.52%
Iteration: 3 | Loss: 2.085792493789702 | Training accuracy: 48.434% | Test accuracy: 54.83%
Iteration: 4 | Loss: 1.8882903026802766 | Training accuracy: 52.774% | Test accuracy: 59.63%
Iteration: 5 | Loss: 1.7602971520875117 | Training accuracy: 55.094% | Test accuracy: 60.21%
Iteration: 6 | Loss: 1.6348791982206847 | Training accuracy: 58.04% | Test accuracy: 62.43%
Iteration: 7 | Loss: 1.5522922139490962 | Training accuracy: 60.128% | Test accuracy: 63.67%
Iteration: 8 | Loss: 1.4837936880948293 | Training accuracy: 62.098% | Test accuracy: 64.31%
Iteration: 9 | Loss: 1.4072782235682164 | Training accuracy: 63.582% | Test accuracy: 66.54%
Iteration: 10 | Loss: 1.3392375324593874 | Training accuracy: 64.368% | Test accuracy: 67.82%
Iteration: 11 | Loss: 1.269238745329752 | Training accuracy: 66.144% | Test accuracy: 68.66%
Iteration: 12 | Loss: 1.2129387459374597 | Training accuracy: 67.342% | Test accuracy: 69.45%
Iteration: 13 | Loss: 1.16239485723947592 | Training accuracy: 68.824% | Test accuracy: 70.31%
Iteration: 14 | Loss: 1.16239485723947592 | Training accuracy: 69.583% | Test accuracy: 71.23%
Application 70046152 exit codes: 1
Application 70046152 resources: utime ~4936s, stime ~1034s, Rss ~1757440, inblocks ~1969380, outblocks ~116314