import csv
import os
import re
import pandas as pd
from ast import literal_eval

DATASET_DIR = 'tweet_data/data_sets'
TRAINING_DF = None
EVAL_DF = None
VALIDATE_DF = None

data_files = [f'tweet_data/tweets/{f}' for f in os.listdir('tweet_data/tweets/') if f != 'README']
max_tweets = 1500



def get_tweets(file, max_tweets=None):
    tweets = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if type(max_tweets) is int:
                if len(tweets) <= max_tweets -1:
                    tweets.append(row[0])
                else:
                    break
            else:
                tweets.append(row[0])

    return tweets

def split_tweets(tweets, split_by=2):
    split_size = len(tweets) // split_by
    batches = []
    start_i = 0

    for i in range(split_by):
        batches.append(tweets[start_i:(start_i + split_size)])
        start_i = start_i + split_size

    print(f"Split tweets into {split_by} batches of {split_size}")
    return batches


def build_datasets(tweet_files):
    training = []
    eval = []
    validate = []

    for f in tweet_files:
        label = f.replace('_tweets.csv', '')
        label = label.replace('tweet_data/tweets/', '')
        tweet_texts = get_tweets(f, max_tweets)
        formatted = [re.sub(r'https?:\/\/\S*[\r\n]*', '', literal_eval(t).decode('utf-8'),
                   flags=re.MULTILINE).replace('&amp', '&') for t in tweet_texts]
        split50 = split_tweets(formatted)
        t_labeled = [(t, label) for t in split50[0]]
        training.extend(t_labeled)
        split25 = split_tweets(split50[1])
        e_labeled = [(t, label) for t in split25[0]]
        eval.extend(e_labeled)
        v_labeled = [(t, label) for t in split25[1]]
        validate.extend(v_labeled)

    training_df = pd.DataFrame({'Tweet':[i[0] for i in training], 'Author': [i[1] for i in training]})
    eval_df = pd.DataFrame({'Tweet':[i[0] for i in eval], 'Author': [i[1] for i in eval]})
    validate_df = pd.DataFrame({'Tweet':[i[0] for i in validate], 'Author': [i[1] for i in validate]})

    return training_df, eval_df, validate_df

sets = build_datasets(data_files)

sets[0].to_csv(f"{DATASET_DIR}/training.csv", sep="\\")
sets[1].to_csv(f"{DATASET_DIR}/eval.csv", sep="\\")
sets[2].to_csv(f"{DATASET_DIR}/validate.csv", sep="\\")
