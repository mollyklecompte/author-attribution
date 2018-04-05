"""
Takes raw tweet text, creates embeddings from character n-grams
"""


import collections
import csv
import os
import re
from ast import literal_eval

import tensorflow as tf


data_files = [f'tweet_data/{f}' for f in os.listdir('tweet_data/')]

# returns list of first 1500 lines of tweet text csv from a file
def get_tweet_text(file):
    tweets = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(tweets) <= 1500:
                tweets.append(row[0])
            else:
                break

    return tweets


# creates character n-grams from string (for single tweet)
def char_ngrammer(input_str, n):
    return [input_str[ind:ind+n] for ind in range(len(input_str)-n +1)]


# returns list of ngram lists from all tweet text
def tweets_to_ngrams(files: list, n):
    ngrams = []
    for f in files:
        tweet_texts = get_tweet_text(f)[:4]
        # creates ngrams for text of DECODED tweet with links removed
        tweet_grams = [(re.sub(r'https?:\/\/\S*[\r\n]*', '', literal_eval(t).decode('utf-8'), flags=re.MULTILINE)) for t in tweet_texts]
        ngrams.extend(tweet_grams)
    return ngrams
