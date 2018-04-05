"""
Takes raw tweet text, creates embeddings from character n-grams
"""


import collections
import csv
import os

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
    return [input_str[ind:ind+n] for ind in range(len(input_str)-n)]


# returns list of ngram lists from all tweet text
def tweets_to_ngrams(files: list, n):
    ngrams = []
    for f in files:
        tweet_texts = get_tweet_text(f)
        # creates ngrams for text of tweet - extra quotes, leading 'b', fixed-len url at end
        tweet_grams = [char_ngrammer(t[2:-25], n) for t in tweet_texts]
        ngrams.extend(tweet_grams)
    return ngrams


