# Part 2 - Recurrent Neural Network

## 2a - Without GloVe Features

Take the following two reviews: * "Although the movie had great visual effects, I thought it was terrible." * "Although the movie had terrible visual effects, I thought it was great."

The first review clearly has an overall negative sentiment while the bottom review clearly has an overall positive sentiment. Both sentences would result in the exact same output if using the bag of words approach.

Clearly there is a lot of useful information which could maybe be utilized more effectively if we didn’t discard the sequence information like we did in part 1. By designing a model capable of capturing this additional source of information, we can potentially achieve better results but also greatly increase the risk of overfitting. This is heavily related to the curse of dimensionality.

A recurrent neural network can be used to maintain the temporal information and process the data as an actual sequence. Part 2 will consist of training recurrent neural networks built with LSTM layers. We will train two separate models again, one from scratch with a word embedding layer and one with GloVe features.