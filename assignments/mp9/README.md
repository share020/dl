# HW9: Human Action Recognition

This assignment will work with the UCF-101 human action recognition dataset. The original technical report published in 2012 discussing the dataset can be found here. The dataset consists of 13,320 videos between ~2-10 seconds long of humans performing one of 101 possible actions. The dimensions of each frame are 320 by 240.

CNNs have been shown to work well in nearly all tasks related to images. A video can be thought of simply as a sequence of images. That is, a network designed for learning from videos must be able to handle the spatial information as well as the temporal information (typically referred to as learning spatiotemporal features). There are many modeling choices to deal with this. Below are a list of papers mentioned in the action recognition presentation given in class. This list is in no way comprehensive but definitely gives a good idea of the general progress.

- Large-scale Video Classification with Convolutional Neural Networks (2014)
- Two-Stream Convolutional Networks for Action Recognition in Videos (2014)
- Beyond Short Snippets: Deep Networks for Video Classification (2015)
- Learning Spatiotemporal Features with 3D Convolutional Networks (2015)
- The Kinetics Human Action Video Dataset (2017)
- Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (2017)
- Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? (2017)

This homework will compare a single frame model (spatial information only) with a 3D convolution based model (2 spatial dimensions + 1 temporal dimension).

- **Part one**: Fine-tune a 50-layer ResNet model (pretrained on ImageNet) on single UCF-101 video frames
- **Part two**: Fine-tune a 50-layer 3D ResNet model (pretrained on Kinetics) on UCF-101 video sequences

The dataset and some pretrained models are located under `/projects/training/bauh/AR`. The original dataset (which can be downloaded here) is relatively small (`9.5GB`). However, it can be extremely slow to load and decode video files from memory on BlueWaters. Because of this, all of the videos have been stored as numpy arrays of size `[sequence_length, height, width, 3]`. This makes them very quick to load but extremely large. The directory `/projects/training/bauh/AR/UCF-101-hdf5` contains all of these files and it is `~572GB`.


## Part 1 - Single Frame Model

This first portion of the homework uses a pretrained ResNet-50 model pretrained on ImageNet. Although the dataset has a large number of frames (13000 videos * number of frames per video), the frames are correlated with eachother meaning there isn’t a whole lot of variety. Also, to keep the sequences relatively short (~2-10 seconds), some of the original videos were split up into 5-6 shorter videos meaning there is even less variety. Single frames alone can still provide a significant amount of information about the action being performed (consider the classes “Skiing” versus “Baseball Pitch”). Training a CNN from scratch significantly overfits. However, the features from a CNN pretrained on ImageNet (over 1 million images of 1000 classes) can be very useful even in video based problem like action recognition. A single frame model performs surprisingly well. This doesn’t necessarily mean solving the task of learning from images inherently solves all video related tasks. It’s more likely that with the problem of human action recognition, the spatial information is more important than the temporal information.


## Part 2 - Sequence Model

Although the single frame model achieves around 73%-75% classification accuracy for a single frame, it above should achieve between 77%-79% after combining the single frame predictions over the whole video. As mentioned above, simply averaging these predictions is a very naive way of classifying sequences. This is similar to the Bag of Words model from the NLP assignment. There are many ways to combine this information in a more intelligent way.

There are many ways to utilize the temporal information. All of the papers in the introduction essentially explore these different techniques. Part 2 of the assignment will do this by using 3D convolutions. 3D convolutions are conceptually the exact same as 2D convolutions except now they also operate over the temporal dimension (sliding window over the frames as well as the image).

The model already overfits on single frames alone. If you were to train a 3D convolutional network from scratch on UCF-101, it severely overfits and has extremely low performance. The Kinetics dataset is a much larger action recognition dataset released more recently than UCF-101. The link above goes to the Kinetics-600 dataset (500,000 videos of 600 various actions). We will use a a 3D ResNet-50 model pretrained on the Kinetics-400 dataset (300,000 videos of 400 various actions) from here. This pretrained model is located in the class directory /projects/training/bauh/AR/ on BlueWaters.


* * *

## Turn in

There are three sets of results for comparison:

1. single-frame model
1. 3D model
1. combined output of the two models.

For each of these three, report the following:

- (`top1_accuracy, top5_accuracy, top10_accuracy`): Did the results improve after combining the outputs?
- Use the confusion matrices to get the 10 classes with the highest performance and the 10 classes with the lowest performance: Are there differences/similarities? Can anything be said about whether particular action classes are discriminated more by spatial information versus temporal information?
- Use the confusion matrices to get the 10 most confused classes. That is, which off-diagonal elements of the confusion matrix are the largest: Are there any notable examples?

Put all of the above into a report and submit as a pdf. Also zip all of the code (not the models, predictions or dataset) and submit.