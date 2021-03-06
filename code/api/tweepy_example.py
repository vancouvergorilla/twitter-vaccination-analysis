from codes import consumer_key, consumer_secret, access_token, access_token_secret
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def on_data(self, data):
        print(data)
        with open("tweets.json", "a") as f:
            f.write(data)
        return True


    def on_error(self, status):
        print(status)

    def on_exception(self, exception):
        print(exception)
        return


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    while True:
        try:
            stream = Stream(auth, l)
    
            stream.filter(track=['covid vaccine', 'covid19 vaccine', 'covid-19 vaccine'], languages=['en'])
        except KeyboardInterrupt:
            break
        except:
            continue
