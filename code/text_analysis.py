import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import random
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(0, 100%, {random.randint(25, 75)}%)"


def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl({random.randint(90, 150)}, 100%, 30%)"


def yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(42, 100%, {random.randint(25, 50)}%)"


def generate_word_cloud(df, sentiment, color_func, tool):
    stopwords = set(STOPWORDS)
    # stopwords.update(["covid", "covid vaccine", "vaccination", "vaccinated", "vaccine", "people"])
    lemmatizer = WordNetLemmatizer()
    raw_text = df[df['sentiment'] == sentiment]['text']
    text = []
    for t in raw_text:
        text += t.split(" ")
    text = [lemmatizer.lemmatize(t) for t in text]
    text = " ".join(t for t in text)

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stopwords).generate(
        text)
    plt.figure()
    plt.imshow(wordcloud.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis("off")
    plt.savefig("../data/results/" + sentiment + "_" + tool + "_wordcloud.png")
    plt.show()


if __name__ == "__main__":
    tool = 'textblob'
    df = pd.read_csv("../data/tweets_sentiment_" + tool + ".csv")
    df = pd.read_csv("../data/tweets_sentiment_" + tool + ".csv")
    generate_word_cloud(df, 'positive', green_color_func, tool)
    generate_word_cloud(df, 'neutral', yellow_color_func, tool)
    generate_word_cloud(df, 'negative', red_color_func, tool)
