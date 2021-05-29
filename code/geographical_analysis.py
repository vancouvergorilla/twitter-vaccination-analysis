import pandas as pd
import json
import csv
from state_dict import us_state_abbrev, states
import numpy as np


def group_sentiment_by_location(tweets_df):  # calculate the average sentiment for each state
    sentiment_by_location = {}
    for index, row in tweets_df.iterrows():
        if row['location'] not in sentiment_by_location:
            sentiment_by_location[row['location']] = {'positive': 0, 'neutral': 0, 'negative': 0, 'sentiment_scores': 0,
                                                      'count': 0}
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
            'avg_sentiment_score': round(
                sentiment_by_location[state]['sentiment_scores'] / sentiment_by_location[state]['count'], 2),
        }
        outputs.append(output)

    outputs = pd.DataFrame(outputs)
    outputs.to_csv('../data/results/sentiment_by_location.csv', index=False, header=True)


def calculate_sentiment_percentage_by_location(tweets_df):
    sentiment_percentage_by_location = {}

    for index, row in tweets_df.iterrows():
        if row['location'] not in sentiment_percentage_by_location:
            sentiment_percentage_by_location[row['location']] = 1
        else:
            sentiment_percentage_by_location[row['location']] += 1

    total_outputs = []
    threshold = 0.021
    other_percentage = 0.0
    for state in sentiment_percentage_by_location.keys():
        percentage = round(sentiment_percentage_by_location[state] / tweets_df.shape[0], 4)
        if percentage < threshold:
            other_percentage += percentage
        else:
            total_output = {
                'location': state,
                'percentage': percentage
            }
            total_outputs.append(total_output)

    total_output = {
        'location': 'Other',
        'percentage': round(other_percentage, 4)
    }
    total_outputs.append(total_output)

    outputs = pd.DataFrame(total_outputs)
    outputs.to_csv('../data/results/sentiment_percentage_by_location.csv', index=False, header=True)


def calculate_sentiment_percentage_by_location_category(tweets_df, category):
    category_sentiment_percentage_by_location = {}
    category_total = 0

    for index, row in tweets_df.iterrows():
        if row['sentiment'] == category:
            category_total += 1
            if row['location'] not in category_sentiment_percentage_by_location:
                category_sentiment_percentage_by_location[row['location']] = 1
            else:
                category_sentiment_percentage_by_location[row['location']] += 1

    outputs = []
    # threshold = 0.021
    # other_percentage = 0.0
    for state in category_sentiment_percentage_by_location.keys():
        percentage = round(category_sentiment_percentage_by_location[state] / category_total, 4)

        output = {
            'location': state,
            'percentage': percentage
        }
        outputs.append(output)

    outputs = pd.DataFrame(outputs)
    outputs.to_csv('../data/results/sentiment_percentage_by_location_' + category + '.csv', index=False, header=True)


def calculate_discussion_rate_by_state(tweets_df):
    state_population = {}

    with open('../data/utils/population.csv', 'r') as file:
        reader = csv.reader(file)
        first_row = True
        for row in reader:
            if first_row:
                first_row = False
                continue
            if row[1] in us_state_abbrev:
                state_population[us_state_abbrev[row[1]]] = int(row[2])

    total_tweets_by_state = {}

    for index, row in tweets_df.iterrows():
        if row['location'] not in total_tweets_by_state:
            total_tweets_by_state[row['location']] = 1
        else:
            total_tweets_by_state[row['location']] += 1

    outputs = []
    min_val = float('inf')
    max_val = float('-inf')
    for state in total_tweets_by_state.keys():
        discussion_rate = total_tweets_by_state[state] / state_population[state]
        if discussion_rate > max_val:
            max_val = discussion_rate
        if discussion_rate < min_val:
            min_val = discussion_rate

        total_tweets_by_state[state] = discussion_rate

    quantile_cal = []

    for state in total_tweets_by_state.keys():
        # discussion_rate_norm = (total_tweets_by_state[state] - min_val) / (max_val - min_val)
        discussion_rate_norm = total_tweets_by_state[state]*100000
        quantile_cal.append(discussion_rate_norm)
        output = {
            "location": state,
            "discussion_rate_norm": discussion_rate_norm
        }
        outputs.append(output)

    print(np.quantile(quantile_cal, 0.25))
    print(np.quantile(quantile_cal, 0.5))
    print(np.quantile(quantile_cal, 0.75))
    outputs = pd.DataFrame(outputs)
    outputs.to_csv('../data/results/discussion_rate_by_state.csv', index=False, header=True)


def calculate_positive_rate_by_location(tweets_df):
    positive_tweets_by_state = {}
    total_tweets_by_state = {}

    for index, row in tweets_df.iterrows():
        if row['location'] not in total_tweets_by_state:
            total_tweets_by_state[row['location']] = 1
        else:
            total_tweets_by_state[row['location']] += 1

        if row['sentiment'] == 'positive' or row['sentiment'] == 'neutral':
            if row['location'] not in positive_tweets_by_state:
                positive_tweets_by_state[row['location']] = 1
            else:
                positive_tweets_by_state[row['location']] += 1

    outputs = []

    for state in total_tweets_by_state.keys():
        positive = 0
        if state in positive_tweets_by_state:
            positive = positive_tweets_by_state[state]

        output = {
            "location": state,
            "positive_rate": round(positive / total_tweets_by_state[state], 4)
        }
        outputs.append(output)

    outputs = pd.DataFrame(outputs)
    outputs.to_csv('../data/results/sentiment_positive_rate_by_state.csv', index=False, header=True)


if __name__ == "__main__":
    df = pd.read_csv("../data/tweets_sentiment_textblob.csv")
    group_sentiment_by_location(df)
    calculate_sentiment_percentage_by_location(df)
    calculate_discussion_rate_by_state(df)
    calculate_positive_rate_by_location(df)
