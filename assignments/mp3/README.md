# HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset.

> The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation.
> 
> For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation.
> 
> For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture.
>
> Due September 28 at 5:00 PM.



## Implementation

### Model Architecture




### Hyper-parameters


|    Hyper-parameters    	|         Description        	|
|:----------------------:	|:--------------------------:	|
|       lr = 0.001       	|        learning rate       	|
|        wd = 5e-4       	|        weight decay        	|
|       epochs = 70      	|       training epochs      	|
| batch\_size_train = 128 	| batch size of training set 	|
|  batch\_size_test = 64  	|   batch size of test set   	|


### Loss function and Optimizer

- Loss funtion: [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)
- Optimizer: [torch.optim.RMSprop](https://pytorch.org/docs/stable/optim.html?highlight=rmsprop#torch.optim.RMSprop)



### Result

- Adam

```
$ python3 main.py
==> Data Augmentation ...
==> Preparing CIFAR10 dataset ...
Files already downloaded and verified
Files already downloaded and verified
==> Initialize CNN model ...
==> Building new CNN model ...
==> Start training ...
Training iteration: 1 | Loss: 1.5323889981145444 | Training accuracy: 57.582% | Test accuracy: 56.94%
==> Saving model ...
Training iteration: 2 | Loss: 1.0421151851144288 | Training accuracy: 65.088% | Test accuracy: 64.73%
Training iteration: 3 | Loss: 0.8457517165052312 | Training accuracy: 72.624% | Test accuracy: 70.18%
Training iteration: 4 | Loss: 0.7301152282206299 | Training accuracy: 76.964% | Test accuracy: 75.73%
Training iteration: 5 | Loss: 0.6541929283105504 | Training accuracy: 79.444% | Test accuracy: 77.75%
Training iteration: 6 | Loss: 0.6053130389631861 | Training accuracy: 81.21% | Test accuracy: 77.82%
Training iteration: 7 | Loss: 0.5568247298755304 | Training accuracy: 81.68% | Test accuracy: 79.69%
Training iteration: 8 | Loss: 0.52486945441007 | Training accuracy: 83.646% | Test accuracy: 80.73%
Training iteration: 9 | Loss: 0.4971955487185427 | Training accuracy: 83.458% | Test accuracy: 80.6%
Training iteration: 10 | Loss: 0.4710853740077494 | Training accuracy: 85.372% | Test accuracy: 82.03%
Training iteration: 11 | Loss: 0.4472699467178501 | Training accuracy: 84.04% | Test accuracy: 80.45%
Training iteration: 12 | Loss: 0.43344204893807314 | Training accuracy: 85.534% | Test accuracy: 81.45%
Training iteration: 13 | Loss: 0.41458634952145157 | Training accuracy: 87.286% | Test accuracy: 83.81%
Training iteration: 14 | Loss: 0.40417786453233656 | Training accuracy: 87.408% | Test accuracy: 83.95%
Training iteration: 15 | Loss: 0.3919767485097851 | Training accuracy: 87.632% | Test accuracy: 84.45%
Training iteration: 16 | Loss: 0.38060398681846724 | Training accuracy: 87.962% | Test accuracy: 84.4%
Training iteration: 17 | Loss: 0.3765119556956889 | Training accuracy: 88.126% | Test accuracy: 83.74%
Training iteration: 18 | Loss: 0.3386370863603509 | Training accuracy: 89.328% | Test accuracy: 84.72%
Training iteration: 19 | Loss: 0.3283725881286899 | Training accuracy: 89.348% | Test accuracy: 85.64%
Training iteration: 20 | Loss: 0.31794708426041374 | Training accuracy: 89.94% | Test accuracy: 85.46%
Training iteration: 21 | Loss: 0.3099242390497871 | Training accuracy: 89.884% | Test accuracy: 85.86%
Training iteration: 22 | Loss: 0.306663738111096 | Training accuracy: 90.072% | Test accuracy: 85.18%
Training iteration: 23 | Loss: 0.29838040646384745 | Training accuracy: 90.27% | Test accuracy: 85.77%
Training iteration: 24 | Loss: 0.2974954615239902 | Training accuracy: 90.752% | Test accuracy: 85.79%
Training iteration: 25 | Loss: 0.2925728217834402 | Training accuracy: 90.312% | Test accuracy: 84.82%
Training iteration: 26 | Loss: 0.2845494441897668 | Training accuracy: 91.51% | Test accuracy: 86.29%
Training iteration: 27 | Loss: 0.27851614777160727 | Training accuracy: 91.114% | Test accuracy: 85.61%
Training iteration: 28 | Loss: 0.2783391854304182 | Training accuracy: 90.108% | Test accuracy: 85.45%
Training iteration: 29 | Loss: 0.2784235685530221 | Training accuracy: 90.908% | Test accuracy: 85.92%
Training iteration: 30 | Loss: 0.27332179284537844 | Training accuracy: 91.622% | Test accuracy: 86.86%
Training iteration: 31 | Loss: 0.26298992027102225 | Training accuracy: 91.722% | Test accuracy: 85.76%
Training iteration: 32 | Loss: 0.2599798788499954 | Training accuracy: 91.376% | Test accuracy: 85.69%
Training iteration: 33 | Loss: 0.25894931983917263 | Training accuracy: 91.72% | Test accuracy: 85.59%
Training iteration: 34 | Loss: 0.2549937529789517 | Training accuracy: 92.59% | Test accuracy: 86.84%
Training iteration: 35 | Loss: 0.2572657864378846 | Training accuracy: 91.864% | Test accuracy: 85.66%
Training iteration: 36 | Loss: 0.25173514092441107 | Training accuracy: 92.08% | Test accuracy: 86.24%
Training iteration: 37 | Loss: 0.24927788639388732 | Training accuracy: 91.174% | Test accuracy: 85.56%
Training iteration: 38 | Loss: 0.24359948014664223 | Training accuracy: 91.89% | Test accuracy: 86.18%
Training iteration: 39 | Loss: 0.24330893989719088 | Training accuracy: 91.932% | Test accuracy: 85.81%
Training iteration: 40 | Loss: 0.2397751587888469 | Training accuracy: 92.314% | Test accuracy: 86.2%
Training iteration: 41 | Loss: 0.238324614341759 | Training accuracy: 91.85% | Test accuracy: 86.08%
Training iteration: 42 | Loss: 0.24105099491451099 | Training accuracy: 91.928% | Test accuracy: 86.0%
Training iteration: 43 | Loss: 0.23592464890702605 | Training accuracy: 92.296% | Test accuracy: 85.67%
Training iteration: 44 | Loss: 0.23125881768401016 | Training accuracy: 92.01% | Test accuracy: 85.88%
Training iteration: 45 | Loss: 0.22848401388243947 | Training accuracy: 93.162% | Test accuracy: 87.34%
Training iteration: 46 | Loss: 0.22769540380638884 | Training accuracy: 92.756% | Test accuracy: 86.17%
Training iteration: 47 | Loss: 0.22363920161105177 | Training accuracy: 92.352% | Test accuracy: 86.12%
Training iteration: 48 | Loss: 0.22299208850278268 | Training accuracy: 93.486% | Test accuracy: 86.86%
Training iteration: 49 | Loss: 0.22150263645688592 | Training accuracy: 92.578% | Test accuracy: 86.6%
Training iteration: 50 | Loss: 0.223169733172335 | Training accuracy: 91.788% | Test accuracy: 86.16%
Training iteration: 51 | Loss: 0.2241714222504355 | Training accuracy: 93.1% | Test accuracy: 86.82%
==> Saving model ...
Training iteration: 52 | Loss: 0.21908805309735296 | Training accuracy: 92.622% | Test accuracy: 87.43%
Training iteration: 53 | Loss: 0.22072116489453084 | Training accuracy: 93.44% | Test accuracy: 86.9%
Training iteration: 54 | Loss: 0.21635687779015897 | Training accuracy: 92.54% | Test accuracy: 86.44%
Training iteration: 55 | Loss: 0.21950814571908062 | Training accuracy: 92.348% | Test accuracy: 85.79%
Training iteration: 56 | Loss: 0.2159109424675822 | Training accuracy: 93.508% | Test accuracy: 86.93%
Training iteration: 57 | Loss: 0.21192567698333575 | Training accuracy: 93.418% | Test accuracy: 87.18%
Training iteration: 58 | Loss: 0.21296871674563878 | Training accuracy: 93.478% | Test accuracy: 87.24%
Training iteration: 59 | Loss: 0.21153371205643925 | Training accuracy: 93.754% | Test accuracy: 87.34%
Training iteration: 60 | Loss: 0.21222362855968574 | Training accuracy: 92.93% | Test accuracy: 85.86%
Training iteration: 61 | Loss: 0.20870164570296207 | Training accuracy: 92.498% | Test accuracy: 86.51%
Training iteration: 62 | Loss: 0.21015779104302912 | Training accuracy: 92.91% | Test accuracy: 86.36%
Training iteration: 63 | Loss: 0.20486266531831468 | Training accuracy: 93.146% | Test accuracy: 87.06%
Training iteration: 64 | Loss: 0.20721056550512534 | Training accuracy: 92.972% | Test accuracy: 87.09%
Training iteration: 65 | Loss: 0.2011126854059184 | Training accuracy: 93.788% | Test accuracy: 86.96%
Training iteration: 66 | Loss: 0.20017541058914132 | Training accuracy: 93.404% | Test accuracy: 86.71%
Training iteration: 67 | Loss: 0.20333445058835437 | Training accuracy: 93.918% | Test accuracy: 87.34%
Training iteration: 68 | Loss: 0.20105336539809357 | Training accuracy: 93.52% | Test accuracy: 86.39%
Training iteration: 69 | Loss: 0.20030613959102375 | Training accuracy: 93.608% | Test accuracy: 87.24%
Training iteration: 70 | Loss: 0.19524423560827894 | Training accuracy: 93.408% | Test accuracy: 87.01%
```

- RMSProp

```
$ python3 main.py
==> Data Augmentation ...
==> Downloading CIFAR10 dataset ...
Files already downloaded and verified
Files already downloaded and verified
==> Initialize CNN model ...
==> Building new CNN model ...
==> Start training ...
[1] loss: 1.798
Accuracy of the network on the test images: 47 %
==>  Saving model..
[2] loss: 1.341
Accuracy of the network on the test images: 56 %
[3] loss: 1.062
Accuracy of the network on the test images: 61 %
[4] loss: 0.886
Accuracy of the network on the test images: 65 %
[5] loss: 0.776
Accuracy of the network on the test images: 73 %
[6] loss: 0.698
Accuracy of the network on the test images: 77 %
[7] loss: 0.641
Accuracy of the network on the test images: 75 %
[8] loss: 0.591
Accuracy of the network on the test images: 78 %
[9] loss: 0.551
Accuracy of the network on the test images: 75 %
[10] loss: 0.521
Accuracy of the network on the test images: 78 %
[11] loss: 0.492
Accuracy of the network on the test images: 80 %
[12] loss: 0.467
Accuracy of the network on the test images: 80 %
[13] loss: 0.449
Accuracy of the network on the test images: 80 %
[14] loss: 0.430
Accuracy of the network on the test images: 81 %
[15] loss: 0.410
Accuracy of the network on the test images: 81 %
[16] loss: 0.396
Accuracy of the network on the test images: 79 %
[17] loss: 0.382
Accuracy of the network on the test images: 82 %
[18] loss: 0.371
Accuracy of the network on the test images: 82 %
[19] loss: 0.357
Accuracy of the network on the test images: 81 %
[20] loss: 0.348
Accuracy of the network on the test images: 80 %
[21] loss: 0.333
Accuracy of the network on the test images: 81 %
[22] loss: 0.327
Accuracy of the network on the test images: 83 %
[23] loss: 0.316
Accuracy of the network on the test images: 83 %
[24] loss: 0.309
Accuracy of the network on the test images: 82 %
[25] loss: 0.299
Accuracy of the network on the test images: 81 %
[26] loss: 0.290
Accuracy of the network on the test images: 82 %
[27] loss: 0.283
Accuracy of the network on the test images: 83 %
[28] loss: 0.278
Accuracy of the network on the test images: 79 %
[29] loss: 0.272
Accuracy of the network on the test images: 83 %
[30] loss: 0.266
Accuracy of the network on the test images: 82 %
[31] loss: 0.260
Accuracy of the network on the test images: 80 %
[32] loss: 0.254
Accuracy of the network on the test images: 84 %
[33] loss: 0.246
Accuracy of the network on the test images: 81 %
[34] loss: 0.244
Accuracy of the network on the test images: 83 %
[35] loss: 0.241
Accuracy of the network on the test images: 82 %
[36] loss: 0.234
Accuracy of the network on the test images: 83 %
[37] loss: 0.234
Accuracy of the network on the test images: 81 %
[38] loss: 0.227
Accuracy of the network on the test images: 83 %
[39] loss: 0.222
Accuracy of the network on the test images: 84 %
[40] loss: 0.218
Accuracy of the network on the test images: 82 %
[41] loss: 0.217
Accuracy of the network on the test images: 85 %
[42] loss: 0.210
Accuracy of the network on the test images: 83 %
[43] loss: 0.209
Accuracy of the network on the test images: 84 %
[44] loss: 0.201
Accuracy of the network on the test images: 83 %
[45] loss: 0.199
Accuracy of the network on the test images: 84 %
[46] loss: 0.197
Accuracy of the network on the test images: 84 %
[47] loss: 0.190
Accuracy of the network on the test images: 82 %
[48] loss: 0.192
Accuracy of the network on the test images: 84 %
[49] loss: 0.189
Accuracy of the network on the test images: 84 %
[50] loss: 0.186
Accuracy of the network on the test images: 81 %
[51] loss: 0.182
Accuracy of the network on the test images: 82 %
==>  Saving model..
[52] loss: 0.183
Accuracy of the network on the test images: 84 %
[53] loss: 0.181
Accuracy of the network on the test images: 84 %
[54] loss: 0.175
Accuracy of the network on the test images: 84 %
[55] loss: 0.172
Accuracy of the network on the test images: 84 %
[56] loss: 0.171
Accuracy of the network on the test images: 81 %
[57] loss: 0.168
Accuracy of the network on the test images: 84 %
[58] loss: 0.168
Accuracy of the network on the test images: 83 %
[59] loss: 0.162
Accuracy of the network on the test images: 83 %
[60] loss: 0.163
Accuracy of the network on the test images: 83 %
[61] loss: 0.157
Accuracy of the network on the test images: 83 %
[62] loss: 0.157
Accuracy of the network on the test images: 84 %
[63] loss: 0.153
Accuracy of the network on the test images: 81 %
[64] loss: 0.154
Accuracy of the network on the test images: 82 %
[65] loss: 0.154
Accuracy of the network on the test images: 84 %
[66] loss: 0.148
Accuracy of the network on the test images: 84 %
[67] loss: 0.146
Accuracy of the network on the test images: 84 %
[68] loss: 0.147
Accuracy of the network on the test images: 84 %
[69] loss: 0.143
Accuracy of the network on the test images: 83 %
[70] loss: 0.142
Accuracy of the network on the test images: 84 %
```