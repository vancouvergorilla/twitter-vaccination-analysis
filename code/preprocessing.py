import json
import preprocessor as p
from datetime import datetime
from state_dict import us_state_abbrev, states
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
from dateutil.parser import parse


def process_location(raw_location):
    if raw_location is None:  # location doesn't exist
        return None
    if raw_location in us_state_abbrev:  # example: Illinois
        return us_state_abbrev[raw_location]
    if raw_location.upper() in states:
        return raw_location.upper()

    if raw_location.count(", ") == 1:
        first_loc, second_loc = raw_location.split(", ")
        if second_loc in us_state_abbrev:  # example: Chicago, Illinois
            return us_state_abbrev[second_loc]
        if second_loc.upper() in states:  # example: Chicago, IL
            return second_loc.upper()
        if first_loc in us_state_abbrev:  # example: Illinois, USA
            return us_state_abbrev[first_loc]

        # remove characters afterward
        second_loc.replace(".", "")
        second_loc.replace(",", "")
        second_loc = second_loc.split(" ")[0]
        if second_loc in us_state_abbrev:
            return us_state_abbrev[second_loc]
        if second_loc.upper() in states:
            return second_loc.upper()

    if raw_location.count(", ") == 2:
        first_loc, second_loc, third_loc = raw_location.split(", ")
        if second_loc in us_state_abbrev:  # example: Chicago, Illinois, USA
            return us_state_abbrev[second_loc]
        if second_loc.upper() in states:  # example: Chicago, IL, USA
            return second_loc.upper()

    return None


def de_emojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


def process_text(raw_text):
    # library preprocessing
    processed_text = p.clean(raw_text)

    # remove stopwords
    processed_text = remove_stopwords(processed_text)

    # convert to lower case
    processed_text = processed_text.lower()

    # remove punctuation
    processed_text = processed_text.replace('[^\\w\\s]', ' ')

    # replace extra white space
    processed_text = processed_text.replace('\\s\\s+', ' ')

    return processed_text


def preprocessing():
    tweets_file = open('../data/tweets.json', )
    tweets_cleaned = []
    original_tweet_count = 0
    exclude_retweet = 0
    passed_tweet = 0

    for line in tweets_file:
        original_tweet_count += 1
        tweet = json.loads(line)

        # exclude retweet
        if 'RT @' in tweet['text']:
            continue

        if datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y') >= datetime.strptime("2021-05-28", "%Y-%m-%d"):
            continue

        exclude_retweet += 1

        # include only tweets in US with state level location
        processed_location = process_location(tweet['user']['location'])
        if processed_location is None:
            continue

        passed_tweet += 1

        raw_text = tweet['text']
        if 'extended_tweet' in tweet:
            raw_text = tweet['extended_tweet']['full_text']
        tweet_cleaned = {
            'date': datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
            'text': process_text(raw_text),
            'location': processed_location,
            'user_name': tweet['user']['name'],
            'user_followers': tweet['user']['followers_count'],
            'user_friends': tweet['user']['friends_count'],
            'user_favorites': tweet['user']['favourites_count'],
        }
        tweets_cleaned.append(tweet_cleaned)

    print(original_tweet_count)
    print(exclude_retweet)
    print(passed_tweet)

    tweets_df = pd.DataFrame(tweets_cleaned)
    tweets_df.sort_values(by='date').reset_index(drop=True)
    tweets_df.to_csv('../data/tweets_cleaned.csv', index=False, header=True)


def preprocessing_kaggle():
    raw_df = pd.read_csv("../data/vaccination_all_tweets.csv")
    original_tweet_count = 0
    passed_tweet = 0
    tweets_cleaned = []

    for index, row in raw_df.iterrows():
        original_tweet_count += 1

        if row['is_retweet']:
            continue

        if isinstance(row['user_location'], float):
            continue

        # include only tweets in US with state level location
        processed_location = process_location(row['user_location'])

        if processed_location is None:
            continue

        tweet_cleaned = {
            'date': row['date'],
            'text': process_text(row['text']),
            'location': processed_location,
            'user_name': row['user_name'],
            'user_followers': row['user_followers'],
            'user_friends': row['user_friends'],
            'user_favorites': row['user_favourites'],
        }

        tweets_cleaned.append(tweet_cleaned)

        passed_tweet += 1

    print(original_tweet_count)
    print(passed_tweet)

    tweets_df = pd.DataFrame(tweets_cleaned)
    tweets_df.sort_values(by='date').reset_index(drop=True)
    tweets_df.to_csv('../data/tweets_cleaned.csv', index=False, header=True)


if __name__ == "__main__":
    preprocessing()
    # preprocessing_kaggle()
