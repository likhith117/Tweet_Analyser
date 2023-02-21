import twint
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import re
from nltk.corpus import stopwords
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer

c = twint.Config()

#query = "#"+input("Enter Hashtag to be analysed: ")
c.Search = "#pawankalyan"
#c.Username="likhith_117"
c.language = "en"
c.Store_object = True
c.Pandas = True

c.Pandas_clean = True
c.Store_csv = False
c.Limit=100
c.Columns = ["username", "tweet"]
#rm lanugyag
twint.run.Search(c)
tweet= twint.storage.panda.Tweets_df[["username", "tweet", "language","date"]]

# c.Output = "N"=
# selected_columns = ['username', 'tweet', 'language']
# print(selected_columns)
# tweet = twint.storage.panda.Tweets_df[selected_columns]

# Select specific columns from the DataFrame


# Access a specific column

# print(tweet.head())


def clean_tweet(tweet):
    # Remove URLs
    
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"https?://\S+|www\.\S+", "", tweet)
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    # Remove usernames
    tweet = re.sub(r"@[A-Za-z0-9_]+", "", tweet)
    tweet = re.sub(r".@", "", tweet)
    tweet = re.sub("[^0-9 ]+", "", tweet)
    # Remove hashtags
    tweet = re.sub(r"#", "", tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    tweet = tweet.lower()
    # Tokenize
    tokens = nltk.word_tokenize(tweet)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    tweet = " ".join(filtered_tokens)
    # Handle negation
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    tweet = " ".join(stemmed_tokens)
    tweet = re.sub(r"\b(?:not|never|no)\b[\w\s]+[^\w\s]", lambda match: re.sub(r'(\s+)(\S+)', r'\1NOT_\2', match.group(0)), tweet)
    return tweet
tweet['clean_tweet'] = tweet['tweet'].apply(clean_tweet)




from nltk.sentiment import SentimentIntensityAnalyzer

# create an instance of the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(tweet):
    # analyze the sentiment of the tweet using VADER
    tweet = clean_tweet(tweet)
    scores = analyzer.polarity_scores(tweet)
    
    # extract the polarity and subjectivity scores from the VADER output
    polarity = scores['compound']
    subjectivity = 1 - scores['neu']
    
    return pd.Series([polarity, subjectivity])

tweet[['polarity', 'subjectivity']] = tweet['clean_tweet'].apply(get_sentiment)


def get_sentiment(tweet):
    if isinstance(tweet, str):
        # analyze the sentiment of the tweet using VADER
        scores = analyzer.polarity_scores(tweet)
        
        # extract the compound score from the VADER output
        compound = scores['compound']
        
        # classify the sentiment based on the compound score
        if compound > 0.05:
            return 'positive'
        elif compound < -0.05:
            return 'negative'
        else:
            return 'neutral'
    else:
        return None   

# Apply the get_sentiment function to each tweet in the dataframe
tweet['sentiment'] = tweet['tweet'].apply(get_sentiment)

# Print the count of each sentiment category



sentiment_counts = tweet['sentiment'].value_counts()

# Create bar plot
sentiment_counts.plot.bar()
plt.title('Sentiment Counts')
plt.show()

# import the necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))

stopwords_set = str(stopwords_set)
stopwords_set = re.sub(r'[^\x00-\x7F]+', '', stopwords_set)
stopwords_set = re.sub(r'@\S+', '', stopwords_set)
stopwords_set = set(stopwords.words('english'))
stopwords_set.update(['https', 'co', 'amp', 'rt','co','to','the'])





# create a string of all the tweets with positive sentiment
positive_tweets = ' '.join(tweet[tweet['sentiment'] == 'positive']['tweet'])

# create a string of all the tweets with negative sentiment
negative_tweets = ' '.join(tweet[tweet['sentiment'] == 'negative']['tweet'])

# # create a WordCloud object for positive tweets
# positive_wordcloud = WordCloud(width = 800, height = 800, 
#                 background_color ='white', 
#                 stopwords = stopwords, 
#                 min_font_size = 10).generate(positive_tweets) 

# # create a WordCloud object for negative tweets
# negative_wordcloud = WordCloud(width = 800, height = 800, 
#                 background_color ='white', 
#                 stopwords = stopwords, 
#                 min_font_size = 10).generate(negative_tweets)



# Generate the wordcloud from the positive tweets
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords_set, 
                min_font_size = 10).generate(positive_tweets) 



# plot the WordCloud image                        
plt.figure(figsize = (10, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',
                stopwords = stopwords_set, 
                min_font_size = 10).generate(negative_tweets)

# plot the WordCloud image                        
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
  



