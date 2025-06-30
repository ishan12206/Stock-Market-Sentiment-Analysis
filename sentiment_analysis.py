import tweepy
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import re
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------- Configuration ----------
BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ---------- Twitter Data Collection ----------
def fetch_tweets(stock_symbol, days=3, max_results=300):
    query = f'${stock_symbol} lang:en -is:retweet'
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    tweets = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=['created_at', 'text'],
        start_time=start_time.isoformat("T") + "Z",
        end_time=end_time.isoformat("T") + "Z",
        max_results=100
    ).flatten(limit=max_results)

    data = [[tweet.created_at, tweet.text] for tweet in tweets]
    return pd.DataFrame(data, columns=['created_at', 'text'])

# ---------- Text Cleaning ----------
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\$\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- Sentiment Scoring ----------
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def analyze_sentiment_with_score(stock_symbol):
    df = fetch_tweets(stock_symbol)
    if df.empty:
        print(f"No tweets found for ${stock_symbol}")
        return pd.DataFrame()

    df['clean_text'] = df['text'].apply(clean_tweet)
    df['polarity'] = df['clean_text'].apply(get_sentiment_score)
    df['date'] = df['created_at'].dt.date
    sentiment_daily = df.groupby('date')['polarity'].mean().reset_index()
    sentiment_daily.columns = ['date', 'avg_sentiment']
    return sentiment_daily

# ---------- Stock Price ----------
def get_stock_price(symbol, days=3):
    end = datetime.today()
    start = end - timedelta(days=days + 2)
    data = yf.download(symbol, start=start, end=end)
    data = data.reset_index()[['Date', 'Close']]
    data['Date'] = data['Date'].dt.date
    data.columns = ['date', 'close']
    return data

# ---------- Correlation & Plot ----------
def correlate_sentiment_with_price(symbol):
    sentiment = analyze_sentiment_with_score(symbol)
    price = get_stock_price(symbol)

    if sentiment.empty or price.empty:
        print("Insufficient data for correlation.")
        return pd.DataFrame()

    df = pd.merge(sentiment, price, on='date', how='inner')
    correlation = df['avg_sentiment'].corr(df['close'])
    print(f"\nCorrelation between sentiment and price for ${symbol}: {correlation:.4f}")

    sns.regplot(data=df, x='avg_sentiment', y='close', line_kws={"color": "red"})
    plt.title(f'Sentiment vs Closing Price for ${symbol}')
    plt.xlabel("Average Sentiment Polarity")
    plt.ylabel("Closing Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df

# ---------- Main ----------
if __name__ == "__main__":
    symbol = "AAPL"  # Change to any stock symbol like TSLA, NVDA, etc.
    result_df = correlate_sentiment_with_price(symbol)
    print("\nMerged Data:")
    print(result_df)
