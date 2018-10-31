# HW8: Sentiment Analysis for IMDB Movie Reviews

> **Part 3 - Language Model**


This section will be about training a language model by leveraging the additional unlabeled data and showing how this pretraining stage typically leads to better results on other tasks like sentiment analysis.

A language model gives some probability distribution over the words in a sequence. We can essentially feed sequences into a recurrent neural network and train the model to predict the following word. Note that this doesn’t require any additional data labeling. The words themselves are the labels. This means we can utilize all `75000` reviews in the training set.

