from textblob import TextBlob
import pandas as pd
import json


def calculate_sentiment(tweets_df):
    tweets_df['sentiment_score'] = tweets_df['text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    tweets_df['sentiment'] = tweets_df['sentiment_score'].apply(
        lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral'))
    print(tweets_df['sentiment'].value_counts())
    tweets_df.to_csv('../data/tweets_sentiment_textblob.csv', index=False, header=True)

    return tweets_df


def group_sentiment_by_location(tweets_df):
    sentiment_by_location = {}
    for index, row in tweets_df.iterrows():
        if row['location'] not in sentiment_by_location:
            sentiment_by_location[row['location']] = {'positive': 0, 'neutral': 0, 'negative': 0, 'sentiment_scores': 0, 'count': 0}
        sentiment_by_location[row['location']][row['sentiment']] += 1
        sentiment_by_location[row['location']]['count'] += 1
        sentiment_by_location[row['location']]['sentiment_scores'] += row['sentiment_score']

    outputs = []
    for state in sentiment_by_location.keys():
        output = {
            'location': state,
            'positive': sentiment_by_location[state]['positive'],
            'neutral': sentiment_by_location[state]['neutral'],
            'negative': sentiment_by_location[state]['negative'],
            'avg_sentiment_score': round(sentiment_by_location[state]['sentiment_scores'] / sentiment_by_location[state]['count'], 2),
        }
        outputs.append(output)

    outputs = pd.DataFrame(outputs)
    outputs.to_csv('../data/sentiment_by_location.csv', index=False, header=True)


if __name__ == "__main__":
    df = pd.read_csv("../data/tweets_cleaned.csv")
    df_sentiment = calculate_sentiment(df)
    group_sentiment_by_location(df_sentiment)
