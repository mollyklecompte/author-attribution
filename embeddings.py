"""
Takes raw tweet text, creates embeddings from character n-grams
"""


import collections
import csv
import os
import re
import gensim
import pandas as pd
from ast import literal_eval



data_files = [f'tweet_data/{f}' for f in os.listdir('tweet_data/') if f != 'README']
DIM_EMBED = 300
INPUT_LEN = 280
NGRAM = 2
window = 5




# creates character n-grams from string (for single tweet)
def char_ngrammer(input_str, n):
    padded_str = input_str+ '\0' * (INPUT_LEN - len(input_str))

    return [padded_str[ind:ind+n] for ind in range(len(padded_str)-n +1)]


# returns list of ngram lists from all tweet text
def tweets_to_ngrams(n):
    ngrams = []
    df = pd.read_csv('tweet_data/data_sets/training.csv', sep="\\")
    tweets = df['Tweet'].tolist()
    for t in tweets:
        if type(t) is not str:
            t = str(t)
        ngrams.append(char_ngrammer(t, n))

    return ngrams

def create_embeddings(ngrams):
    model = gensim.models.Word2Vec(ngrams, size=DIM_EMBED, window=window, min_count=0, iter=10)

    return model

model = create_embeddings(tweets_to_ngrams(NGRAM))
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

