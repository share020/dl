# HW7: Sentiment Analysis for IMDB Movie Reviews

> Due November 2

## Overview

This assignment will work with the Large Movie Review Dataset and provide an understanding of how Deep Learning is used within the field of Natural Language Processing (NLP). The original paper published in 2011 discussing the dataset and a wide variety of techniques can be found here. We will train models to detect the sentiment of written text. More specifically, we will try to feed a movie review into a model and have it predict whether it is a positive or negative movie review.

The assignment starts off with how to preprocess the dataset into a more appropriate format followed by three main parts. Each part will involve training different types of models and will detail what you need to do as well as what needs to be turned in. The assignment is meant to provide you with a working knowledge of how to use neural networks with sequential data.

- Part one deals with training basic Bag of Words models.
- Part two will start incoporating temporal information by using LSTM layers.
- Part three will show how to train a language model and how doing this as a first step can sometimes improve results for other tasks

Make a new directory for yourself on BlueWaters. Do everything within this directory and this will ultimately contain most of what you need to turn in.

I'm not directly providing you with any finished python files. This is written more like a tutorial with excerpts of code scattered throughout. You will have to take these pieces of code, combine them, edit them, add to them, etc. and figure out how to run them on BlueWaters. You will also need to change a variety of hyperparameters and run them multiple times.

### What to Turn In

Make a copy of the google sheet here. This is how you should summarize your results. Write a sentence or two for each model you train in parts 1a, 1b, 2a, 2b, and 3c. For part 3a, the assignment only requires you to train the model with the hyperparamters I give you. For part 3b, you will generate fake movie reviews (I've included a fake movie review I generated as an example).

Zip all of your code (excluding the dataset/preprocessed data/trained PyTorch models). Save your completed google sheet as a PDF. Submit the code and PDF on compass.


### Some Basic Terminology

One-hot vector - Mathematically speaking, a one-hot vector is a vector of all zeros except for one element containing a one. There are many applications but they are of particular importance in natural language processing. It is a method of encoding categorical variables (such as words or characters) as numerical vectors which become the input to computational models.

Suppose we want to train a model to predict the sentiment of the following two sentences: “This movie is bad” and “This movie is good”. We need a way of mathematically representing these as inputs to the model. First we define a dictionary (or vocabulary). This contains every word contained within the dataset.


```python
vocab = ['this','movie','is','bad','good']
```


We can also add an “unknown token” to the dictionary in case words not present in the training dataset show up during testing.

```python
unknown = ''
vocab = [unknown,'this','movie','is','bad','good']
```


Each word can now be represented as vector of length |vocab| with a 1 in the location corresponding to the word's location within the dictionary.

```python
[0,1,0,0,0,0] # this
[0,0,1,0,0,0] # movie
[0,0,0,1,0,0] # is
[0,0,0,0,0,1] # good
```

There are many ways this data can be used. This assignment deals with two possibilities. These vectors could be processed one at a time (which is the basis of sections 2 and 3 of this assignment) by using a recurrent neural network or the full review could be processed all at once using a bag of words representation (section 1 of this assignment). The bag of words representation simply sums up the one-hot vectors and has a single vector input for each review. This is convenient for reducing the input size and making variable length reviews fixed length (all reviews become a single vector of length |vocab|) but the input no longer contains sequence information. The order in which the words appear has no effect on a document's bag of words representation.

```python
x1=[0,1,1,1,0,1] # bag of words for "this movie is good"
x2=[0,1,1,1,1,0] # bag of words for "this movie is bad"
```

A basic classifier would be to multiply the bag of words representation x by some weight vector W and declare the review to be positive if the output is greater than 0 and negative otherwise. Training this model on these two sentences might yield a weight vector `W=[0.0,0.0,0.0,0.0,+0.5,-0.5]`. Notice this model simply classifies any review with the word “good” as positive and any review with the word “bad” as negative.


```python
# model would successfully classify the unseen test sequence as negative
x=[7,0,0,0,1,0] # bag of words for "I have never seen such a bad film"

# model would incorrectly classify the unseen test sequence as negative
x=[10,1,1,1,1,0] # bag of words for "This movie is not bad. In fact, I think it's great." Notice the symbols "." and "," are treated as unknown tokens.
```

Word Embedding - The one-hot encoding is useful but notice the input size depends heavily on the vocabulary size which can cause problems considering there are millions of tokens in the English language. Additionally, many of these tokens are related to eachother in some way (like the words "good" and "great") and this relationship is not captured by the one-hot encoding. Word embeddings are another vector representation of fixed length (for example, length `300` is common) where the elements are real valued as opposed to just `1` or `0`. Typically these word embeddings are generated from a language model which is the basis of part 3 of this assignment. Word2vec and GloVe are examples of word embeddings.

## Preprocessing the Data

The IMDB dataset is located within the class directory under `/projects/training/bauh/NLP/aclImdb/`. Each review is contained in a separate `.txt` file and organized into separate directories depending on if they're positive/negative or part of the train/test/unlabeled splits. It's easier to re-work the data before training any models for multiple reasons.

It's faster to combine everything into fewer files.
We need to tokenize the dataset (split long strings up into individual words/symbols).
We need to know how many unique tokens there are to decide on our vocabulary/model size.
We don't want to constantly be converting strings to token IDs within the training loop (takes non-negligible amount of time).
Storing everything as token IDs directly leads to the 1-hot representation necessary for input into our model.


The NLTK (Natural Language Toolkit) python package contains a tokenizer that we will use and needs to be installed.

```
pip install --user nltk
```

Within your assignment directory, create a file called `preprocess_data.py`.


```python
import numpy as np
import os
import nltk
import itertools
import io

# create directory to store preprocessed data
if(not os.path.isdir('preprocessed_data')):
    os.mkdir('preprocessed_data')
```

These few lines of code written above import a few packages and creates a new directory if it's not already made.


```python
# get all of the training reviews (including unlabeled reviews)
train_directory = '/projects/training/bauh/NLP/aclImdb/train/'

pos_filenames = os.listdir(train_directory + 'pos/')
neg_filenames = os.listdir(train_directory + 'neg/')
unsup_filenames = os.listdir(train_directory + 'unsup/')

pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]
unsup_filenames = [train_directory+'unsup/'+filename for filename in unsup_filenames]

filenames = pos_filenames + neg_filenames + unsup_filenames

count = 0
x_train = []
for filename in filenames:
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_train.append(line)
    count += 1
    print(count)
```


The top part of the code simply results in a list of every text file containing a movie review in the variable filenames. The first 12500 are positive reviews, the next 12500 are negative reviews, and the remaining are unlabeled reviews. For each review, we remove text we don't want ('' and ‘\x96'), tokenize the string using the nltk package, and make everything lowercase. Here is an example movie review before and after tokenization:

```
"`Mad Dog' Earle is back, along with his sad-sack moll Marie, and that fickle clubfoot Velma. So are Babe and Red, Doc and Big Mac, and even the scenery-chewing mutt Pard. The only thing missing is a good reason for remaking Raoul Walsh's High Sierra 14 years later without rethinking a line or a frame, and doing so with talent noticeably a rung or two down the ladder from that in the original. (Instead of Walsh we get Stuart Heisler, for Humphrey Bogart we get Jack Palance, for Ida Lupino Shelley Winters, and so on down through the credits.) The only change is that, this time, instead of black-and-white, it's in Warnercolor; sadly, there are those who would count this an improvement.<br /><br />I Died A Thousand Times may be unnecessary \x96 and inferior \x96 but at least it's not a travesty; the story still works on its own stagy terms. Earle (Palance), fresh out of the pen near Chicago, drives west to spearhead a big job masterminded by ailing kingpin Lon Chaney, Jr. \x96 knocking over a post mountain resort. En route, he almost collides with a family of Oakies, when he's smitten with their granddaughter; the smiting holds even when he discovers she's lame. Arriving at the cabins where the rest of gang holes up, he finds amateurish hotheads at one another's throats as well as Winters, who throws herself at him (as does the pooch). Biding time until they get a call from their inside man at the hotel, Palance (to Winter's chagrin) offers to pay for an operation to cure the girl's deformity, a gesture that backfires. Then, the surgical strike against the resort turns into a bloodbath. On the lam, Palance moves higher into the cold Sierras....<br /><br />It's an absorbing enough story, competently executed, that lacks the distinctiveness Walsh and his cast brought to it in 1941, the year Bogie, with this role and that of Sam Spade in the Maltese Falcon, became a star. And one last, heretical note: Those mountains do look gorgeous in color.<br /><br />"
```


```
['`', 'mad', 'dog', "'", 'earle', 'is', 'back', ',', 'along', 'with', 'his', 'sad-sack', 'moll', 'marie', ',', 'and', 'that', 'fickle', 'clubfoot', 'velma', '.', 'so', 'are', 'babe', 'and', 'red', ',', 'doc', 'and', 'big', 'mac', ',', 'and', 'even', 'the', 'scenery-chewing', 'mutt', 'pard', '.', 'the', 'only', 'thing', 'missing', 'is', 'a', 'good', 'reason', 'for', 'remaking', 'raoul', 'walsh', "'s", 'high', 'sierra', '14', 'years', 'later', 'without', 'rethinking', 'a', 'line', 'or', 'a', 'frame', ',', 'and', 'doing', 'so', 'with', 'talent', 'noticeably', 'a', 'rung', 'or', 'two', 'down', 'the', 'ladder', 'from', 'that', 'in', 'the', 'original', '.', '(', 'instead', 'of', 'walsh', 'we', 'get', 'stuart', 'heisler', ',', 'for', 'humphrey', 'bogart', 'we', 'get', 'jack', 'palance', ',', 'for', 'ida', 'lupino', 'shelley', 'winters', ',', 'and', 'so', 'on', 'down', 'through', 'the', 'credits', '.', ')', 'the', 'only', 'change', 'is', 'that', ',', 'this', 'time', ',', 'instead', 'of', 'black-and-white', ',', 'it', "'s", 'in', 'warnercolor', ';', 'sadly', ',', 'there', 'are', 'those', 'who', 'would', 'count', 'this', 'an', 'improvement', '.', 'i', 'died', 'a', 'thousand', 'times', 'may', 'be', 'unnecessary', 'and', 'inferior', 'but', 'at', 'least', 'it', "'s", 'not', 'a', 'travesty', ';', 'the', 'story', 'still', 'works', 'on', 'its', 'own', 'stagy', 'terms', '.', 'earle', '(', 'palance', ')', ',', 'fresh', 'out', 'of', 'the', 'pen', 'near', 'chicago', ',', 'drives', 'west', 'to', 'spearhead', 'a', 'big', 'job', 'masterminded', 'by', 'ailing', 'kingpin', 'lon', 'chaney', ',', 'jr.', 'knocking', 'over', 'a', 'post', 'mountain', 'resort', '.', 'en', 'route', ',', 'he', 'almost', 'collides', 'with', 'a', 'family', 'of', 'oakies', ',', 'when', 'he', "'s", 'smitten', 'with', 'their', 'granddaughter', ';', 'the', 'smiting', 'holds', 'even', 'when', 'he', 'discovers', 'she', "'s", 'lame', '.', 'arriving', 'at', 'the', 'cabins', 'where', 'the', 'rest', 'of', 'gang', 'holes', 'up', ',', 'he', 'finds', 'amateurish', 'hotheads', 'at', 'one', 'another', "'s", 'throats', 'as', 'well', 'as', 'winters', ',', 'who', 'throws', 'herself', 'at', 'him', '(', 'as', 'does', 'the', 'pooch', ')', '.', 'biding', 'time', 'until', 'they', 'get', 'a', 'call', 'from', 'their', 'inside', 'man', 'at', 'the', 'hotel', ',', 'palance', '(', 'to', 'winter', "'s", 'chagrin', ')', 'offers', 'to', 'pay', 'for', 'an', 'operation', 'to', 'cure', 'the', 'girl', "'s", 'deformity', ',', 'a', 'gesture', 'that', 'backfires', '.', 'then', ',', 'the', 'surgical', 'strike', 'against', 'the', 'resort', 'turns', 'into', 'a', 'bloodbath', '.', 'on', 'the', 'lam', ',', 'palance', 'moves', 'higher', 'into', 'the', 'cold', 'sierras', '...', '.', 'it', "'s", 'an', 'absorbing', 'enough', 'story', ',', 'competently', 'executed', ',', 'that', 'lacks', 'the', 'distinctiveness', 'walsh', 'and', 'his', 'cast', 'brought', 'to', 'it', 'in', '1941', ',', 'the', 'year', 'bogie', ',', 'with', 'this', 'role', 'and', 'that', 'of', 'sam', 'spade', 'in', 'the', 'maltese', 'falcon', ',', 'became', 'a', 'star', '.', 'and', 'one', 'last', ',', 'heretical', 'note', ':', 'those', 'mountains', 'do', 'look', 'gorgeous', 'in', 'color', '.']
```

Notice how symbols like periods, parenthesis, etc. become their own tokens and it splits up contractions (“it's” becomes “it” and “'s”). It's not perfect and can be ruined by typos or lack of punctuation but works for the most part. We now have a list of tokens for every review in the training dataset in the variable `x_train`. We can do the same thing for the test dataset and the variable `x_test`.

 
```
# get all of the test reviews
test_directory = '/projects/training/bauh/NLP/aclImdb/test/'

pos_filenames = os.listdir(test_directory + 'pos/')
neg_filenames = os.listdir(test_directory + 'neg/')

pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]

filenames = pos_filenames+neg_filenames

count = 0
x_test = []
for filename in filenames:
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_test.append(line)
    count += 1
    print(count)
```

We can use the following to get a basic understanding what the IMDB dataset contains.


```
# number of tokens per review
no_of_tokens = []
for tokens in x_train:
    no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)
print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))
```


> `Total: 20090526 Min: 10 Max: 2859 Mean: 267.87368 Std: 198.540647165`


The mean review contains ~267 tokens with a standard deviation of ~200. Although there are over 20 million total tokens, they're obviously not all unique. We now want to build our dictionary/vocabulary.


```
# word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)
```


We have two more variables: `word_to_id` and `id_to_word`. You can access `id_to_word[index]` to find out the token for a given index or access `word_to_id[token]` to find out the `index` for a given token. They'll both be used later on. We can also see there are roughly 200,000 unique tokens with `len(id_to_word)`. Realistically, we don't want to use all 200k unique tokens. An embedding layer with 200k tokens and 500 outputs has 100 million weights which is far too many considering the training dataset only has 20 million tokens total. Additionally, the tokenizer sometimes creates a unique token that is actually just a combination of other tokens because of typos and other odd things. Here are a few examples from `id_to_word` ("majesty.these", "producer/director/star", "1¢")


We should organize the tokens by their frequency in the dataset to give us a better idea of choosing our vocabulary size.


```
# let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
    print(id_to_word[i],count[i])
```


Notice we used `word_to_id` to convert all of the tokens for each review into a sequence of token IDs instead (this is ultimately what we are going for but we want the IDs to be in a different order). We then accumulate the total number of occurrences for each word in the array `count`. Finally, this is used to sort the vocabulary list. We are left with `id_to_word` in order from most frequent tokens to the least frequent, `word_to_id` gives us the index for each token, and count simply contains the number of occurrences for each token.


```python
(array([164036,  26990,   7825,   1367]), array([    1,    10,   100,  1000, 10000]))
the 1009387
, 829574
. 819080
and 491830
a 488305
of 438545
to 405661
is 330904
it 285710
in 280618
```


The histogram output gives us a better understanding of the actual dataset. Over 80% (~160k) of the unique tokens occur between 1 and 10 times while only ~5% occur more than 100 times each. Using `np.sum(count[0:100])` tells us over half of all of the 20 million tokens are the most common 100 words and `np.sum(count[0:8000])` tells us almost 95% of the dataset is contained within the most common 8000 words.


```python
# assign -1 if token doesn't appear in our dictionary
# add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
```


Here is where we convert everything to the exact format we want for training purposes. Notice the test dataset may have unique tokens our model has never seen before. We can anticipate this ahead of time by actually reserving index 0 for an unknown token. This is why I assign a -1 if the token isn't part of `word_to_id` and add `+1` to every id. Just be aware of this that `id_to_word` is now off by `1` index if you actually want to convert ids to words.

We could also decide later on if we only want to use a vocabulary size of 8000 for training and just assign any other token ID in the training data to 0. This way it can develop its own embedding for unknown tokens which can help out when it inevitably sees unknown tokens during testing.

Here is the same review posted earlier. One uses the full dictionary and the other replaces the less frequent tokens with the unknown token:

```python
[   727   1203    844     85  23503      8    156      2    351     17
     32  50333   9855   2299      2      4     13  21676 159129  26371
      3     44     30   4381      4    819      2   3430      4    212
   6093      2      4     66      1  26454  24093 184660      3      1
     73    162    957      8      5     58    296     18  12511  11708
   5009     14    337  10898   3188    164    313    215  50604      5
    372     48      5   2147      2      4    397     44     17    637
  12385      5  25011     48    119    204      1   7427     43     13
     10      1    214      3     23    304      6   5009     80     92
   5002 185207      2     18   5281   2982     80     92    668   6463
      2     18  12795  10997   5805   6830      2      4     44     27
    204    155      1    893      3     22      1     73    634      8
     13      2     12     68      2    304      6   8241      2      9
     14     10 112033    122   1056      2     46     30    157     42
     63   1558     12     41   4960      3     11   1077      5   3317
    223    211     34   1750      4   4583     20     37    228      9
     14     29      5   4826    122      1     74    145    513     27
    105    206  16280   1262      3  23503     23   6463     22      2
   1431     54      6      1   8616    782   3312      2   3121   1283
      7  40800      5    212    309  41644     40  13044  10740   5693
   3349      2   2251   6817    142      5   2177   2680   4480      3
   6543   4548      2     31    229  28016     17      5    239      6
 111582      2     60     31     14   9966     17     72   9632    122
      1  83734   1736     66     60     31   1993     62     14    838
      3   7842     37      1  36053    124      1    375      6   1122
   1500     64      2     31    620   2462  92441     37     35    165
     14  10636     16     86     16   6830      2     42   2721    783
     37    101     23     16     82      1  24765     22      3  47291
     68    357     38     92      5    639     43     72    979    139
     37      1   1392      2   6463     23      7   3818     14  14581
     22   1510      7    967     18     41   4229      7   4470      1
    249     14  25202      2      5   9040     13  19413      3    106
      2      1  17660   3097    438      1   4480    503     94      5
  11569      3     27      1  11129      2   6463   1091   1924     94
      1   1125  57215     67      3      9     14     41   6813    200
     74      2  10368   2102      2     13   1407      1  55353   5009
      4     32    186    788      7      9     10   7493      2      1
    342  13056      2     17     12    220      4     13      6   1039
   9839     10      1  12443   9205      2    859      5    348      3
      4     35    235      2  42426    885     83    157   4577     50
    175   1497     10   1360      3]
```

```python
[ 727 1203  844   85    0    8  156    2  351   17   32    0    0 2299    2
    4   13    0    0    0    3   44   30 4381    4  819    2 3430    4  212
 6093    2    4   66    1    0    0    0    3    1   73  162  957    8    5
   58  296   18    0    0 5009   14  337    0 3188  164  313  215    0    5
  372   48    5 2147    2    4  397   44   17  637    0    5    0   48  119
  204    1 7427   43   13   10    1  214    3   23  304    6 5009   80   92
 5002    0    2   18 5281 2982   80   92  668 6463    2   18    0    0 5805
 6830    2    4   44   27  204  155    1  893    3   22    1   73  634    8
   13    2   12   68    2  304    6    0    2    9   14   10    0  122 1056
    2   46   30  157   42   63 1558   12   41 4960    3   11 1077    5 3317
  223  211   34 1750    4 4583   20   37  228    9   14   29    5 4826  122
    1   74  145  513   27  105  206    0 1262    3    0   23 6463   22    2
 1431   54    6    1    0  782 3312    2 3121 1283    7    0    5  212  309
    0   40    0    0 5693 3349    2 2251 6817  142    5 2177 2680 4480    3
 6543 4548    2   31  229    0   17    5  239    6    0    2   60   31   14
    0   17   72    0  122    1    0 1736   66   60   31 1993   62   14  838
    3 7842   37    1    0  124    1  375    6 1122 1500   64    2   31  620
 2462    0   37   35  165   14    0   16   86   16 6830    2   42 2721  783
   37  101   23   16   82    1    0   22    3    0   68  357   38   92    5
  639   43   72  979  139   37    1 1392    2 6463   23    7 3818   14    0
   22 1510    7  967   18   41 4229    7 4470    1  249   14    0    2    5
    0   13    0    3  106    2    1    0 3097  438    1 4480  503   94    5
    0    3   27    1    0    2 6463 1091 1924   94    1 1125    0   67    3
    9   14   41 6813  200   74    2    0 2102    2   13 1407    1    0 5009
    4   32  186  788    7    9   10 7493    2    1  342    0    2   17   12
  220    4   13    6 1039    0   10    1    0    0    2  859    5  348    3
    4   35  235    2    0  885   83  157 4577   50  175 1497   10 1360    3]
```

To finish up, we save our token id based reviews to text files and save our dictionary in case we ever need a conversion of ID back to text.


```python
# save dictionary
np.save('preprocessed_data/imdb_dictionary.npy',np.asarray(id_to_word))

# save training data to single text file
with io.open('preprocessed_data/imdb_train.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

# save test data to single text file
with io.open('preprocessed_data/imdb_test.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
```


### GloVe Features

The GloVe features already provide a text file with a 300 dimensional word embedding for over 2 million tokens. This dictionary is sorted by word frequency as well but for a different dataset. Therefore, we still need to have some way of using the IMDB dataset with their dictionary. Continue adding this code to preprocess_data.py


```python
glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'
with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))
```

We have two new arrays `glove_dictionary` and `glove_embeddings`. The first is the same as `id_to_word` but a different order and `glove_embeddings` contain the actual embeddings for each token. To save space, only the first 100k tokens are kept. Also, notice a 300 dimensional vector of 0s is prepended to the array of embeddings to be used for the `unknown` token.


```
word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
```

We do exactly what we did before but with this new GloVe dictionary and finally save the data in the new format.


```python
np.save('preprocessed_data/glove_dictionary.npy',glove_dictionary)
np.save('preprocessed_data/glove_embeddings.npy',glove_embeddings)

with io.open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

with io.open('preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
```
