import re
from datetime import datetime
import nltk
import pandas as pd
import tweepy as tw
from deep_translator import GoogleTranslator
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from functions import twitter_access, twitter_api, tweets_fetching, text_cleaning, translator, sentiment_determination
import os
from dotenv import load_dotenv

###----###
def sentiment(data):
    start = datetime.now()

    # twitter_account = pd.read_csv('Twitter Developer Account.csv')
    # consumer_api_key, consumer_api_key_secret, access_token, access_token_secret = twitter_access(twitter_account)
    consumer_api_key = os.getenv("CONSUMER_API_KEY")
    consumer_api_key_secret = os.getenv("CONSUMER_API_KEY_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    api = twitter_api(consumer_api_key, consumer_api_key_secret, access_token, access_token_secret)

    raw_tweets_list = tweets_fetching(api, data, 100, since_date='2022-05-28')
    tweets_df = pd.DataFrame({
        'raw' : raw_tweets_list,
    })

    tweets_df['clean'] = tweets_df['raw'].apply(lambda x: text_cleaning(x))
    tweets_df['translate'] = tweets_df['clean'].apply(lambda x: translator(x))
    tweets_df = sentiment_determination(tweets_df)

    print("Sentiment in general: ", tweets_df['sentiment'].value_counts().index[0])

    end = datetime.now()
    print("Excecution time: ", (end-start).seconds)
    return tweets_df