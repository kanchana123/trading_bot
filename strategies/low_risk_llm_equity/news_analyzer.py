# Note: This module requires third-party libraries. Please install them using:
# pip install gnews nltk
#
# Additionally, the VADER lexicon from NLTK is required. This module will
# attempt to download it automatically if it's not found.

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from gnews import GNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class NewsAnalyzer:
    """
    Analyzes news sentiment for a given stock symbol.
    Can fetch historical or real-time news and returns an overall sentiment
    and the top headlines.
    """

    def __init__(self):
        """
        Initializes the NewsAnalyzer and the VADER sentiment analyzer.
        """
        try:
            self._analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            print("Downloading VADER lexicon for NLTK...")
            import nltk

            nltk.download("vader_lexicon")
            self._analyzer = SentimentIntensityAnalyzer()

        self._gnews = GNews(language="en", country="IN", max_results=10)

    def get_news_sentiment(
        self, stock_symbol: str, analysis_date: Optional[datetime] = None
    ) -> Dict:
        """
        Gets news headlines and analyzes their sentiment.

        If analysis_date is provided, it fetches historical news for that day.
        If analysis_date is None, it fetches real-time news from the last 24 hours.

        Args:
            stock_symbol (str): The stock symbol to search news for (e.g., 'AAPL').
            analysis_date (Optional[datetime], optional): The date for historical
                                                         analysis. Defaults to None.

        Returns:
            Dict: A dictionary containing:
                  - 'overall_sentiment': 'Positive', 'Negative', or 'Neutral'.
                  - 'sentiment_score': The average compound sentiment score.
                  - 'top_headlines': A list of the top 3 most significant headlines.
        """
        if analysis_date:
            # Historical analysis for a specific day
            end_date = analysis_date
            start_date = end_date - timedelta(days=1)
        else:
            # Real-time analysis for the last 24 hours
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

        headlines = self._fetch_headlines(stock_symbol, start_date, end_date)

        if not headlines:
            return {
                "overall_sentiment": "Neutral",
                "sentiment_score": 0.0,
                "top_headlines": [],
                "note": "No recent news headlines found.",
            }

        return self._analyze_sentiment(headlines)

    def _fetch_headlines(
        self, stock_symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        """
        Fetches news headlines for a stock within a date range using GNews.
        """
        self._gnews.start_date = (start_date.year, start_date.month, start_date.day)
        self._gnews.end_date = (end_date.year, end_date.month, end_date.day)

        # Use a more specific query for better results
        query = f'"{stock_symbol}" stock OR market news'
        try:
            news = self._gnews.get_news(query)
            return news
        except Exception as e:
            print(f"Error fetching news from GNews: {e}")
            return []

    def _analyze_sentiment(self, headlines: List[Dict]) -> Dict:
        """
        Analyzes sentiment of headlines, calculates an overall score, and finds
        the most significant articles.
        """
        analyzed_articles = []
        total_compound_score = 0.0

        for article in headlines:
            title = article.get("title", "")
            sentiment = self._analyzer.polarity_scores(title)
            compound_score = sentiment.get("compound", 0.0)

            analyzed_articles.append({"title": title, "url": article.get("url"), "compound_score": compound_score})
            total_compound_score += compound_score

        analyzed_articles.sort(key=lambda x: abs(x["compound_score"]), reverse=True)
        top_headlines = [{"title": a["title"], "url": a["url"]} for a in analyzed_articles[:3]]
        average_score = total_compound_score / len(headlines) if headlines else 0.0

        overall_sentiment = "Neutral"
        if average_score >= 0.05: overall_sentiment = "Positive"
        elif average_score <= -0.05: overall_sentiment = "Negative"

        return {"overall_sentiment": overall_sentiment, "sentiment_score": round(average_score, 4), "top_headlines": top_headlines} 