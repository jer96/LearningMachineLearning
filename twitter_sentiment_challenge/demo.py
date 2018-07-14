import tweepy
from textblob import TextBlob
import numpy as np 
import pandas as pd 

# Step 1 - Authenticate
consumer_key= ''
consumer_secret= ''

access_token=''
access_token_secret=''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('Trump')

#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself
tweets = []

for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)

    sent = analysis.sentiment
    sub = analysis.subjectivity

    # only consider tweets that are highly subjective
    if(sub > .5):
        tweet_tensor = None
        # positive threshold > .7
        if(sent > .7):
            tweet_tensor = [tweet, 'positive']
        # negative threshold < .3
        elif(sent < .3):
            tweet_tensor = [tweet, 'negative']
        else:
            tweet_tensor = [tweet, 'neutral']
    tweets.append(tweet_tensor)

columns = ['tweet', 'sentiment']
df = pd.DataFrame(tweets, columns=columns)
df.to_csv('sentiment_challenge.csv', index=False)