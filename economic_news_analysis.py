# economic_news_analysis.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

class EconomicUncertaintyAnalyzer:
    """
    Class for analyzing economic uncertainty and news sentiment.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.epu_data = None
        self.tariff_data = None
        self.news_sentiment = None

        # Load environment variables from .env file
        load_dotenv()

        # Access your API key
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        
        # Download required NLTK data if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def load_economic_data(self, epu_file='data/epu_index.csv', tariff_file='data/tariff_impact.csv'):
        """
        Load economic policy uncertainty data and tariff impact data.
        
        Parameters:
        -----------
        epu_file : str, optional
            Path to economic policy uncertainty file
        tariff_file : str, optional
            Path to tariff impact file
        
        Returns:
        --------
        pd.DataFrame
            Combined economic data
        """
        try:
            # Load EPU data
            self.epu_data = pd.read_csv(epu_file, index_col=0, parse_dates=True)
            print(f"Loaded EPU data with {len(self.epu_data)} rows")
            
            # Load tariff data
            self.tariff_data = pd.read_csv(tariff_file, index_col=0, parse_dates=True)
            print(f"Loaded tariff impact data with {len(self.tariff_data)} rows")
            
            # Combine data
            combined_data = pd.DataFrame(index=self.epu_data.index)
            combined_data['EPU_Index'] = self.epu_data['EPU_Index']
            
            # Reindex tariff data to match EPU dates (forward fill)
            if not self.tariff_data.empty:
                reindexed_tariff = self.tariff_data.reindex(
                    index=pd.date_range(start=self.tariff_data.index.min(), 
                                       end=self.epu_data.index.max(), 
                                       freq='D')
                )
                reindexed_tariff = reindexed_tariff.fillna(method='ffill')
                
                # Add to combined data
                combined_data['Tariff_Impact'] = reindexed_tariff.reindex(combined_data.index)['Tariff_Impact']
            
            return combined_data
        
        except Exception as e:
            print(f"Error loading economic data: {str(e)}")
            return pd.DataFrame()
        
    def analyze_news_sentiment(self, news_data=None, news_file='data/news_articles.csv'):
        """
        Analyze sentiment from news articles related to tariffs and trade.
        
        Parameters:
        -----------
        news_data : pd.DataFrame, optional
            DataFrame containing news articles
        news_file : str, optional
            Path to news articles file if news_data not provided
        
        Returns:
        --------
        pd.DataFrame
            News sentiment analysis results
        """
        try:
            # Load news data if not provided
            if news_data is None:
                news_data = pd.read_csv(news_file, index_col=0, parse_dates=True)
            
            # Initialize sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Define keywords related to tariffs and trade
            tariff_keywords = ['tariff', 'trade war', 'import tax', 'export tax', 
                              'trade agreement', 'trade policy', 'trade tension']
            
            # Filter articles containing tariff keywords
            if 'content' in news_data.columns:
                tariff_news = news_data[news_data['content'].str.contains('|'.join(tariff_keywords), 
                                                                         case=False, regex=True)]
            else:
                print("No 'content' column found in news data.")
                return pd.DataFrame()
            
            # Calculate sentiment for each article
            sentiments = []
            
            for idx, row in tariff_news.iterrows():
                sentiment = sia.polarity_scores(row['content'])
                sentiments.append({
                    'date': idx,
                    'title': row.get('title', ''),
                    'compound': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu']
                })
            
            # Create sentiment DataFrame
            sentiment_df = pd.DataFrame(sentiments)
            if not sentiment_df.empty:
                sentiment_df.set_index('date', inplace=True)
                
                # Calculate daily aggregate sentiment
                daily_sentiment = sentiment_df.resample('D').mean()
                
                # Store results
                self.news_sentiment = daily_sentiment
                
                return daily_sentiment
            else:
                print("No tariff-related articles found.")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error analyzing news sentiment: {str(e)}")
            return pd.DataFrame()
        
    def fetch_news_articles(self, api_key, keywords=['tariff', 'trade war'], days=30):
        """
        Fetch news articles using a news API.
        
        Parameters:
        -----------
        api_key : str
            API key for news service
        keywords : list, optional
            Keywords to search for
        days : int, optional
            Number of days to fetch
        
        Returns:
        --------
        pd.DataFrame
            DataFrame of news articles
        """
        try:
            # Format search query
            query = ' OR '.join(keywords)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Make API request
            url = (f"https://newsapi.org/v2/everything?"
                  f"q={query}&"
                  f"from={from_date}&"
                  f"to={to_date}&"
                  f"language=en&"
                  f"sortBy=relevancy&"
                  f"apiKey={api_key}")
            
            response = requests.get(url)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                # Process articles
                news_data = []
                for article in articles:
                    news_data.append({
                        'date': article.get('publishedAt'),
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })
                
                # Create DataFrame
                news_df = pd.DataFrame(news_data)
                
                # Convert date to datetime
                news_df['date'] = pd.to_datetime(news_df['date'])
                news_df.set_index('date', inplace=True)
                
                # Save to CSV
                os.makedirs('data', exist_ok=True)
                news_df.to_csv('data/news_articles.csv')
                
                print(f"Fetched {len(news_df)} news articles")
                return news_df
            
            else:
                print(f"Error fetching news: {response.status_code}")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return pd.DataFrame()