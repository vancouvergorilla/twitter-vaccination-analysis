
import pandas as pd
import nltk
nltk.download('vader_lexicon') # Download the VADER lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def calculate_sentiment(tweets_df):

    # Initialize sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Obtaining NLTK scores
    tweets_df['sentiment_score'] = tweets_df['text'].apply(lambda x: sia.polarity_scores(x))

    # Obtaining NLTK compound score
    tweets_df['sentiment_cmp_score'] = tweets_df['sentiment_score'].apply(lambda score_dict: score_dict['compound'])


    # Categorize scores into the sentiments of positive, neutral or negative
    tweets_df['sentiment'] = tweets_df['sentiment_cmp_score'].apply(
        lambda score: 'positive' if score > 0 else ('negative' if score < -0 else 'neutral'))
    
    print(tweets_df['sentiment'].value_counts())
    tweets_df.to_csv('../data/tweets_sentiment_NLTK.csv', index=False, header=True)

    return tweets_df



if __name__ == "__main__":
    df = pd.read_csv("../../data/tweets_cleaned.csv")
    df_sentiment = calculate_sentiment(df)
