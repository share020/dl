"""
HW8: Sentiment Analysis for IMDB Movie Reviews.

Part 3 - Language Model
    3b - Generating Fake Reviews

@author: Zhenye Na
@credit: Logan Courtney
"""

import torch
import numpy as np

from torch.autograd import Variable
from RNN_language_model import RNN_language_model

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1

word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}

model = torch.load('language.model')
print('model loaded...')
model.cuda()
model.eval()

# create partial sentences to "prime" the model
# this implementation requires the partial sentences
# to be the same length if doing more than one
# tokens = [['i','love','this','movie','.'],['i','hate','this','movie','.']]
tokens = [['a'], ['i']]

token_ids = np.asarray(
    [[word_to_id.get(token, -1) + 1 for token in x] for x in tokens])

# preload phrase
x = Variable(torch.LongTensor(token_ids)).cuda()

embed = model.embedding(x)  # batch_size, time_steps, features

state_size = [embed.shape[0], embed.shape[2]]  # batch_size, features
no_of_timesteps = embed.shape[1]

model.reset_state()

outputs = []
for i in range(no_of_timesteps):

    h = model.lstm1(embed[:, i, :])
    h = model.bn_lstm1(h)
    h = model.dropout1(h, dropout=0.3, train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h, dropout=0.3, train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h, dropout=0.3, train=False)

    h = model.decoder(h)

    outputs.append(h)

outputs = torch.stack(outputs)
outputs = outputs.permute(1, 2, 0)
output = outputs[:, :, -1]


temperatures = [1.0, 0.5, 0.233]  # float(sys.argv[1])
length_of_review = 150


for temperature in temperatures:
    print("Now trying with temperature: {}".format(temperature))
    review = []
    for j in range(length_of_review):

        # sample a word from the previous output
        output = output / temperature
        probs = torch.exp(output)
        probs[:, 0] = 0.0
        probs = probs / (torch.sum(probs, dim=1).unsqueeze(1))
        x = torch.multinomial(probs, 1)
        review.append(x.cpu().data.numpy()[:, 0])

        # predict the next word
        embed = model.embedding(x)

        h = model.lstm1(embed[:, 0, :])
        h = model.bn_lstm1(h)
        h = model.dropout1(h, dropout=0.3, train=False)

        h = model.lstm2(h)
        h = model.bn_lstm2(h)
        h = model.dropout2(h, dropout=0.3, train=False)

        h = model.lstm3(h)
        h = model.bn_lstm3(h)
        h = model.dropout3(h, dropout=0.3, train=False)

        output = model.decoder(h)

    review = np.asarray(review)
    review = review.T
    review = np.concatenate((token_ids, review), axis=1)
    review = review - 1
    review[review < 0] = vocab_size - 1
    review_words = imdb_dictionary[review]
    for review in review_words:
        prnt_str = ''
        for word in review:
            prnt_str += word
            prnt_str += ' '
        print(prnt_str)
