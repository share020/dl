# Part 1 - Bag of Words

## 1b - Using GloVe Features

We will now train another bag of words model but use the pre-trained GloVe features in place of the word embedding layer from before. Typically by leveraging larger datasets and regularization techniques, overfitting can be reduced. The GloVe features were pre-trained on over 840 billion tokens. Our training dataset contains 20 million tokens and 2⁄3 of the 20 million tokens are part of the unlabeled reviews which weren’t used in part 1a. The GloVe dataset is over 100 thousand times larger.

The hope is then that these 300 dimensional GloVe features already contain a significant amount of useful information since they were pre-trained on such a large dataset and that will improve performance for our sentiment classification.

