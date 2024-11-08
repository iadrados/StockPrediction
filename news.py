from transformers import pipeline
import yfinance as yf
from goose3 import Goose
from requests import get

def get_ticker_news_sentiment(ticker):
    """
    Returns a Pandas dataframe of the given ticker's most recent news article headlines,
    with the overal sentiment of each article.

    Args:
        ticker (string)

    Returns:
        pd.DataFrame: {'Date', 'Article title', Article sentiment'}
    """
    ticker_news = yf.Ticker(ticker)
    news_list = ticker_news.get_news()
    extractor = Goose()
    pipe = pipeline("text-classification", model="ProsusAI/finbert")

