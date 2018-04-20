
"""
based on https://gist.github.com/yanofsky/5436496
modified to get full text of tweet up to updated 280 char limit
"""


import tweepy
import csv

# Twitter API credentials (I removed my access tokens here too)
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""


def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200, tweet_mode='extended')

    # save most recent tweets
    def add_new_tweets(new):
        newer_tweets = []
        for tweet in new:
            try:
                text = tweet.full_text
            except:
                text = tweet.text
            if 'RT @' not in text:
                newer_tweets.append(tweet)
        alltweets.extend(newer_tweets)
    #

    add_new_tweets(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before tweet ID %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest, tweet_mode='extended')

        # save most recent tweets
        add_new_tweets(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    # transform the tweepy tweets into a 2D array that will populate the csv
    # outtweets = [[tweet.created_at, tweet.full_text.encode("utf-8")] for tweet in alltweets]
    outtweets = []
    for tweet in alltweets:
        try:
            text = tweet.full_text
        except:
            text = tweet.text
        outtweets.append([text.encode()])

    # write the csv
    with open('tweet_data/%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(outtweets)

    pass


if __name__ == '__main__':
    # pass in the username of the account you want to download
    get_all_tweets("")