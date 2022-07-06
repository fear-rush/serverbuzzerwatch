import re
import nltk
import pandas as pd
import tweepy as tw
from deep_translator import GoogleTranslator
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv

nltk.download('vader_lexicon')

def twitter_access(twitter_account):

  # consumer_api_key = twitter_account.loc[0,'Id']
  # consumer_api_key_secret = twitter_account.loc[1,'Id']
  # access_token = twitter_account.loc[2,'Id']
  # access_token_secret = twitter_account.loc[3,'Id']

  consumer_api_key = os.getenv("CONSUMER_API_KEY")
  consumer_api_key_secret = os.getenv("CONSUMER_API_KEY_SECRET")
  access_token = os.getenv("ACCESS_TOKEN")
  access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

  return consumer_api_key, consumer_api_key_secret, access_token, access_token_secret

def twitter_api(consumer_api_key, consumer_api_key_secret, access_token, access_token_secret):

  authenticate = tw.OAuthHandler(consumer_api_key, consumer_api_key_secret)
  authenticate.set_access_token(access_token, access_token_secret)
  api = tw.API(authenticate)

  return api

def tweets_fetching(api, keyword, num_of_tweets, since_date=None, until_date=None):

  raw_tweets_list = []

  tweets = tw.Cursor(api.search, 
                     q=keyword+'-filter:retweets', 
                     lang='id',
                     since=since_date,
                     until=until_date, 
                     tweet_mode='extended').items(num_of_tweets)

  for tweet in tweets:
    raw_tweets_list.append(tweet.full_text)
  
  return raw_tweets_list

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def text_cleaning(text):
  
  text = remove_emojis(text)
  text = text.lower()
  text = text.replace('\n',' ') 
  text = text.replace('’','')
  text = re.sub(r'(@[A-Za-z0-9_]+)', '', text)
  text = re.sub(r'#[A-Za-z0-9_]+', '', text)
  text = re.sub(r'https?:\/\/\S*', '', text)
  text = re.sub(r"[^\w\d'\s]+",'', text)
  text = re.sub(r'[0-9]+', '',  text)
  text = re.sub(' +', ' ', text)
  text = text.strip()

  return text

def translator(text):
  text = GoogleTranslator(source='id', target='en').translate(text)
  return text

def sentiment_determination(tweets_df):

  tweets_df[['polarity', 'subjectivity']] = tweets_df['translate'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

  for index, row in tweets_df['translate'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)

    if score['neg'] > score['pos']:
      tweets_df.loc[index, 'sentiment'] = 'negative'
    elif score['pos'] > score['neg']:
      tweets_df.loc[index, 'sentiment'] = 'positive'
    else:
      tweets_df.loc[index, 'sentiment'] = 'neutral'

    tweets_df.loc[index, 'pos'] = score['pos']
    tweets_df.loc[index, 'neu'] = score['neu']
    tweets_df.loc[index, 'neg'] = score['neg']
    tweets_df.loc[index, 'compound'] = score['compound']

  return tweets_df