
import pandas as pd
import numpy as np
import stanza
stanza.download('en')

nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

def stanza_analyze(Text):
    document = nlp(Text)
    return np.mean([(i.sentiment - 1) for i in document.sentences]) # Minus 1 so as to bring score range of [0,2] to [-1,1]


def calculate_sentiment(tweets_df):

    tweets_df['sentiment_score'] = tweets_df['text'].apply(lambda x: stanza_analyze(x))

    # Convert average Stanza sentiment score into sentiment categories
    tweets_df['sentiment'] = tweets_df['sentiment_score'].apply(
        lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral'))

    
    print(tweets_df['sentiment'].value_counts())
    tweets_df.to_csv('../data/tweets_sentiment_stanza.csv', index=False, header=True)

    return tweets_df



if __name__ == "__main__":
    df = pd.read_csv("../data/tweets_cleaned.csv")
    df_sentiment = calculate_sentiment(df)
