import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Load the CSV dataset into a Pandas DataFrame
df = pd.read_csv(r'C:\Users\Hp\Downloads\socialmedia.csv')#put your path of csv file or dataset here 

# Data Preprocessing
df['Post Text'] = df['Post Text'].str.lower()

# Sentiment Analysis
df['sentiment'] = df['Post Text'].apply(get_sentiment)

# User Engagement Analysis
numerical_engagement_columns = ['User ID', 'Likes/Reactions', 'Shares/Retweets', 'Comments']
df['total_engagement'] = df[numerical_engagement_columns].sum(axis=1)

# Interactive Filtering
sentiment_range = (-1.0, 1.0)
filtered_df = df[(df['sentiment'] >= sentiment_range[0]) & (df['sentiment'] <= sentiment_range[1])]

# Display Data Table
print('Filtered Data:')
print(filtered_df)

# Sentiment Analysis Visualization
plt.figure(figsize=(8, 5))
filtered_df['sentiment'].hist(bins=20, color='blue', alpha=0.7)
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# User Engagement Analysis Visualization
plt.figure(figsize=(8, 5))
filtered_df.plot(y='total_engagement', kind='line', color='green', marker='o')
plt.title('User Engagement Analysis')
plt.ylabel('Total Engagement')
plt.show()
