"""
Takes raw tweet text, creates embeddings from character n-grams
"""


import collections
import csv
import os
import re
import gensim
import numpy as np
from ast import literal_eval

import tensorflow as tf


data_files = [f'tweet_data/{f}' for f in os.listdir('tweet_data/') if f != 'README']
DIM_EMBED = 300
INPUT_LEN = 280
window = 5

# returns list of first 1500 lines of tweet text csv from a file
def get_tweet_text(file):
    tweets = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(tweets) <= 1499:
                tweets.append(row[0])
            else:
                break

    return tweets


# creates character n-grams from string (for single tweet)
def char_ngrammer(input_str, n):
    padded_str = input_str+ '\0' * (INPUT_LEN - len(input_str))
    return [padded_str[ind:ind+n] for ind in range(len(padded_str)-n +1)]


# returns list of ngram lists from all tweet text
def tweets_to_ngrams(files: list, n):
    ngrams = []
    for f in files:
        tweet_texts = get_tweet_text(f)
        # creates ngrams for text of DECODED tweet
        tweet_grams = [char_ngrammer(re.sub(r'https?:\/\/\S*[\r\n]*', '', literal_eval(t).decode('utf-8'), flags=re.MULTILINE).replace('&amp', '&'), n) for t in tweet_texts]
        ngrams.extend(tweet_grams)
    return ngrams

def create_embeddings(ngrams):
    model = gensim.models.Word2Vec(ngrams, size=DIM_EMBED, window=window, min_count=0, iter=10)

    return model


model = create_embeddings(tweets_to_ngrams(data_files, 2))
model.save('models/embeddings')
# print(len(model.wv.vocab))
# print(list(model.wv.vocab.keys()))

# model = gensim.models.Word2Vec.load('models/embeddings')
# embedding_matrix = np.zeros((len(model.wv.vocab), DIM_EMBED))
# for i in range(len(model.wv.vocab)):
#     embedding_vector = model.wv[model.wv.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector


# print(len(model.wv.vocab))
# print(model)
# vocab_len = model.wv.vocab
# initializer = np.zeros((vocab_len, DIM_EMBED), dtype=np.float32)

