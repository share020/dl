# Part 1 - Bag of Words

## 1a - Without GloVe Features

A bag of words model is one of the most basic models for document classification. It is a way of handling varying length inputs (sequence of token ids) to get a single output (positive or negative sentiment).

The token IDs can be thought of as an alternative representation of a 1-hot vector. A word embedding layer is usually a `d` by `V` matrix where `V` is the size of the vocabulary and d is the embedding size. Multiplying a 1-hot vector (of length `V`) by this matrix results in a d dimensional embedding for a particular token. We can avoid this matrix multiplication by simply taking the column of the word embedding matrix corresponding with that particular token.

A word embedding matrix doesn't tell you how to handle the sequence information. In this case, you can picture the bag of words as simply taking the mean of all of the 1-hot vectors for the entire sequence and then multiplying this by the word embedding matrix to get a document embedding. Alternatively, you can take the mean of all of the word embeddings for each token and get the same document embedding. Now we are left with a single `d` dimensional input representing the entire sequence.

Note that the bag of words method utilizes all of the tokens in the sequence but loses all of the information about when the tokens appear within the sequence. Obviously the order of words carries a significant portion of the information contained within written text but this technique still works out rather well.