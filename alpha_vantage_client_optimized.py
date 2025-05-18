# alpha_vantage_client_optimized.py

import requests
import pandas as pd
import numpy as np
import time
import os
import json
import hashlib
import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class AlphaVantageError(Exception):
    """Base exception for Alpha Vantage API errors"""
    pass

class RateLimitError(AlphaVantageError):
    """Raised when API rate limits are exceeded"""
    pass

class InvalidAPICallError(AlphaVantageError):
    """Raised when the API call is invalid"""
    pass

class NotEnoughDataError(AlphaVantageError):
    """Raised when there's not enough data for the requested timeframe"""
    pass

class TokenBucket:
    """
    Implements the token bucket algorithm for rate limiting.
    Tokens are added at a fixed rate, and consumed when making API calls.
    """
    def __init__(self, tokens, fill_rate):
        """
        Initialize the token bucket.
        
        Parameters:
        -----------
        tokens : int
            Maximum number of tokens in the bucket
        fill_rate : float
            Rate at which tokens are added to the bucket (tokens/second)
        """
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_check = time.time()
        self.lock = threading.RLock()
        
    def consume(self, tokens=1):
        """
        Consume tokens from the bucket. Returns True if tokens were consumed,
        False if not enough tokens are available.
        
        Parameters:
        -----------
        tokens : int, optional
            Number of tokens to consume (default: 1)
            
        Returns:
        --------
        bool
            True if tokens were consumed, False otherwise
        """
        with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            if self.tokens < self.capacity:
                delta = self.fill_rate * (now - self.last_check)
                self.tokens = min(self.capacity, self.tokens + delta)
            self.last_check = now
            
            # Check if we have enough tokens and consume them
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False

class TieredCache:
    """
    Implements a tiered caching system with memory and disk layers.
    Different TTLs are set based on data type for optimal freshness.
    """
    def __init__(self, memory_ttl=timedelta(minutes=5), 
                 disk_ttl=timedelta(days=1), 
                 db_path="alpha_vantage_cache.db"):
        """
        Initialize the tiered cache.
        
        Parameters:
        -----------
        memory_ttl : timedelta, optional
            Default time-to-live for memory cache
        disk_ttl : timedelta, optional
            Default time-to-live for disk cache
        db_path : str, optional
            Path to SQLite database for disk cache
        """
        self.memory_cache = {}
        self.memory_ttl = {}
        self.memory_default_ttl = memory_ttl
        self.disk_default_ttl = disk_ttl
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Setup disk cache
        self._setup_db()
        
    def _setup_db(self):
        """Set up SQLite database for disk caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            data TEXT,
            last_updated TIMESTAMP,
            expires_at TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
    
    def _determine_ttl(self, params):
        """
        Determine appropriate TTL based on data type.
        
        Parameters:
        -----------
        params : dict
            API request parameters
            
        Returns:
        --------
        timedelta
            Appropriate TTL for this data type
        """
        function = params.get("function", "")
        
        # Set TTLs based on data type
        if function in ["TIME_SERIES_INTRADAY"]:
            return timedelta(minutes=15)  # Intraday data refreshes frequently
        elif function in ["TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED"]:
            return timedelta(hours=6)  # Daily data refreshed at market close
        elif function in ["TIME_SERIES_WEEKLY", "TIME_SERIES_WEEKLY_ADJUSTED"]:
            return timedelta(days=1)  # Weekly data doesn't change much
        elif function in ["TIME_SERIES_MONTHLY", "TIME_SERIES_MONTHLY_ADJUSTED"]:
            return timedelta(days=7)  # Monthly data changes rarely
        elif function in ["GLOBAL_QUOTE"]:
            return timedelta(minutes=1)  # Real-time quotes need frequent refresh
        elif function in ["COMMODITY", "WTI", "BRENT", "NATURAL_GAS"]:
            return timedelta(hours=12)  # Commodity data updates daily or less
        elif function == "NEWS_SENTIMENT":
            return timedelta(hours=2)  # News sentiment changes throughout the day
        else:
            return self.memory_default_ttl
    
    def _get_cache_key(self, params):
        """
        Generate a unique cache key based on request parameters.
        
        Parameters:
        -----------
        params : dict
            API request parameters
            
        Returns:
        --------
        str
            MD5 hash of sorted parameters
        """
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(self, params):
        """
        Get data from tiered cache system.
        Tries memory first, then disk.
        
        Parameters:
        -----------
        params : dict
            API request parameters
            
        Returns:
        --------
        object or None
            Cached data if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(params)
        now = datetime.now()
        
        with self.lock:
            # First check memory cache
            if cache_key in self.memory_cache:
                if now < self.memory_ttl.get(cache_key, now):
                    return self.memory_cache[cache_key]
                else:
                    # Expired, remove from memory
                    del self.memory_cache[cache_key]
                    if cache_key in self.memory_ttl:
                        del self.memory_ttl[cache_key]
            
            # Then check disk cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT data, expires_at FROM cache WHERE key = ?", 
                          (cache_key,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                data_str, expires_at_str = row
                expires_at = datetime.fromisoformat(expires_at_str)
                
                if now < expires_at:
                    # Still valid, load into memory cache too
                    data = json.loads(data_str)
                    self.memory_cache[cache_key] = data
                    self.memory_ttl[cache_key] = expires_at
                    return data
            
            # Not found in either cache
            return None
    
    def set(self, params, data):
        """
        Store data in both memory and disk caches.
        
        Parameters:
        -----------
        params : dict
            API request parameters
        data : object
            Data to cache
        """
        cache_key = self._get_cache_key(params)
        ttl = self._determine_ttl(params)
        now = datetime.now()
        expires_at = now + ttl
        
        with self.lock:
            # Store in memory cache
            self.memory_cache[cache_key] = data
            self.memory_ttl[cache_key] = expires_at
            
            # Store in disk cache
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (key, data, last_updated, expires_at) VALUES (?, ?, ?, ?)",
                    (cache_key, json.dumps(data), now.isoformat(), expires_at.isoformat())
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"Error storing in disk cache: {str(e)}")

    def clear_expired(self):
        """Clear expired entries from both memory and disk caches."""
        now = datetime.now()
        
        with self.lock:
            # Clear expired memory cache entries
            expired_keys = [k for k, t in self.memory_ttl.items() if now >= t]
            for key in expired_keys:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.memory_ttl:
                    del self.memory_ttl[key]
            
            # Clear expired disk cache entries
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE expires_at < ?", 
                              (now.isoformat(),))
                conn.commit()
                conn.close()
                
                logger.debug(f"Cleared {len(expired_keys)} expired items from memory cache "
                           f"and {cursor.rowcount} from disk cache")
            except Exception as e:
                logger.warning(f"Error clearing expired disk cache entries: {str(e)}")


class AlphaVantageClient:
    """
    Optimized client for accessing AlphaVantage API endpoints with
    tiered caching, intelligent rate limiting, and robust error handling.
    """
    
    def __init__(self, api_key=None, base_url="https://www.alphavantage.co/query", 
                 rate_limit=True, cache_enabled=True, cache_dir="cache"):
        """
        Initialize the AlphaVantage client.
        
        Parameters:
        -----------
        api_key : str, optional
            AlphaVantage API key (if None, looks for env var ALPHA_VANTAGE_API_KEY)
        base_url : str, optional
            Base URL for the AlphaVantage API
        rate_limit : bool, optional
            Whether to apply rate limiting
        cache_enabled : bool, optional
            Whether to use caching
        cache_dir : str, optional
            Directory for cache files
        """
        self.api_key = self._get_api_key(api_key)
        self.base_url = base_url
        
        # Configure rate limiting based on API tier
        self.rate_limit = rate_limit
        # Premium tier still has limits but they're higher
        # Adjust based on your specific premium tier
        self.rate_limiter = TokenBucket(10, 10/60)  # 10 requests per minute
        
        # Setup caching
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.tiered_cache = TieredCache(
            memory_ttl=timedelta(minutes=5),
            disk_ttl=timedelta(hours=24),
            db_path=os.path.join(cache_dir, "alpha_vantage_cache.db")
        )
        
        # Create persistent HTTP session for connection pooling
        self.session = requests.Session()
        
        # Track request statistics
        self.request_count = 0
        self.last_request_time = 0
        self.error_counts = {}
        
        logger.info("AlphaVantage client initialized with premium tier settings")
    
    def _get_api_key(self, api_key):
        """
        Get API key, using environment variable if not provided.
        
        Parameters:
        -----------
        api_key : str or None
            API key or None
            
        Returns:
        --------
        str
            API key
            
        Raises:
        -------
        ValueError
            If no API key is available
        """
        if api_key:
            return api_key
        
        env_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if env_key:
            return env_key
        
        raise ValueError("No AlphaVantage API key provided. Set ALPHA_VANTAGE_API_KEY environment variable.")
    
    def _respect_rate_limit(self):
        """
        Apply token bucket algorithm for rate limiting.
        Blocks until a token is available.
        
        Returns:
        --------
        bool
            True when a token has been consumed
        """
        if not self.rate_limit:
            return True
        
        # Wait for token if needed
        while not self.rate_limiter.consume():
            time.sleep(0.1)
        
        return True
    
    def _validate_response(self, response_data):
        """
        Check if Alpha Vantage API response contains valid data.
        
        Parameters:
        -----------
        response_data : dict
            API response data
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        if not response_data or not isinstance(response_data, dict):
            return False, "Empty or invalid response"
        
        # Check for explicit error messages
        if "Error Message" in response_data:
            error_msg = response_data["Error Message"]
            if "Invalid API call" in error_msg:
                return False, f"Invalid API call: {error_msg}"
            return False, f"API error: {error_msg}"
        
        # Check for rate limiting
        if "Note" in response_data and "call frequency" in response_data["Note"]:
            return False, f"Rate limit exceeded: {response_data['Note']}"
        
        # Check for time series data
        time_series_key = next((k for k in response_data.keys() 
                               if k.startswith("Time Series") or 
                               k.startswith("Monthly") or
                               k.startswith("Weekly") or 
                               k in ["Global Quote", "Meta Data", "data"]),
                              None)
        
        if time_series_key and time_series_key != "Meta Data":
            # Check if we have actual data points
            data_points = response_data.get(time_series_key, {})
            if not data_points or len(data_points) == 0:
                return False, f"No data points available in response"
            return True, "Valid response with data points"
        
        # For other endpoints like currency exchange
        if "Realtime Currency Exchange Rate" in response_data:
            return True, "Valid exchange rate data"
        
        # For commodity data
        if "data" in response_data and isinstance(response_data["data"], list):
            if len(response_data["data"]) > 0:
                return True, "Valid commodity data"
            return False, "Empty commodity data list"
        
        # For news sentiment data
        if "feed" in response_data:
            if response_data["feed"] and len(response_data["feed"]) > 0:
                return True, "Valid news sentiment data"
            return False, "Empty news feed"
        
        # Fallback check - if we have any significant data
        if len(response_data.keys()) > 1:
            return True, "Response appears to contain data"
        
        return False, "No recognizable data structure in response"
    
    def _make_request(self, params, use_cache=True, retry_count=3, backoff_factor=2):
        """
        Make a request to the AlphaVantage API with caching, rate limiting,
        and error handling.
        
        Parameters:
        -----------
        params : dict
            API request parameters
        use_cache : bool, optional
            Whether to use cache
        retry_count : int, optional
            Number of retry attempts
        backoff_factor : int, optional
            Factor for exponential backoff
            
        Returns:
        --------
        dict
            API response
            
        Raises:
        -------
        RateLimitError
            If rate limit is exceeded
        InvalidAPICallError
            If API call is invalid
        AlphaVantageError
            For other API errors
        """
        # Add API key
        complete_params = {**params, "apikey": self.api_key}
        
        # Check cache first if enabled
        if self.cache_enabled and use_cache:
            cached_data = self.tiered_cache.get(params)
            if cached_data:
                logger.debug(f"Cache hit for {params.get('function', 'unknown')}")
                return cached_data
        
        # Apply rate limiting
        self._respect_rate_limit()
        
        # Make the request with retries and exponential backoff
        for attempt in range(retry_count):
            try:
                self.request_count += 1
                self.last_request_time = time.time()
                
                logger.debug(f"Making API request: {params.get('function', 'unknown')} (attempt {attempt+1}/{retry_count})")
                response = self.session.get(self.base_url, params=complete_params)
                response.raise_for_status()
                
                # Parse response
                if params.get('datatype', 'json') == 'csv':
                    data = response.text
                else:
                    data = response.json()
                
                # Validate the response
                is_valid, message = self._validate_response(data)
                if not is_valid:
                    error_type = "RateLimit" if "rate limit" in message.lower() else "InvalidAPI"
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
                    if "rate limit" in message.lower():
                        if attempt < retry_count - 1:
                            wait_time = backoff_factor ** attempt
                            logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry.")
                            time.sleep(wait_time)
                            continue
                        raise RateLimitError(message)
                    elif "invalid api call" in message.lower():
                        raise InvalidAPICallError(message)
                    else:
                        if "no data points" in message.lower():
                            raise NotEnoughDataError(message)
                        raise AlphaVantageError(message)
                
                # Cache successful response
                if self.cache_enabled and use_cache:
                    self.tiered_cache.set(params, data)
                
                return data
                
            except (RateLimitError, InvalidAPICallError, NotEnoughDataError) as e:
                # Pass through specific exceptions
                raise
                
            except requests.RequestException as e:
                logger.warning(f"Request failed: {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise AlphaVantageError(f"Request failed after {retry_count} attempts: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise AlphaVantageError(f"Unexpected error: {str(e)}")
        
        # Should never reach here, but just in case
        raise AlphaVantageError("Max retries exceeded")
    
    def get_daily_adjusted(self, symbol, outputsize="full", cache=True, cache_days=1, fallback=True):
        """
        Get daily adjusted time series for a symbol with enhanced reliability.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        outputsize : str, optional
            'compact' for latest 100 data points, 'full' for up to 20 years of data
        cache : bool, optional
            Whether to use cached data
        cache_days : int, optional
            Number of days before cache expires
        fallback : bool, optional
            Whether to try alternative approaches if the request fails
            
        Returns:
        --------
        pd.DataFrame
            Daily adjusted time series data
        """
        # Define cache file path for backwards compatibility with old cache
        cache_file = f"{self.cache_dir}/{symbol}_daily_adjusted.csv"
        
        # Check if cache file exists and is recent
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < cache_days * 86400:
                logger.info(f"Using cached daily data for {symbol}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Prepare API parameters
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize,
            'datatype': 'json'
        }
        
        try:
            # Make the request
            data = self._make_request(params, use_cache=cache)
            
            # Parse the time series data
            time_series_key = 'Time Series (Daily)'
            
            if time_series_key not in data:
                raise NotEnoughDataError(f"No time series data found for {symbol}")
            
            time_series = data.get(time_series_key, {})
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns and convert to numeric
            df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 
                          'volume', 'dividend_amount', 'split_coefficient']
            df = df.apply(pd.to_numeric)
            
            # Calculate returns
            df['returns'] = df['adjusted_close'].pct_change()
            
            # Cache the result if requested
            if cache:
                df.to_csv(cache_file)
            
            return df
            
        except NotEnoughDataError as e:
            logger.warning(f"Not enough data for {symbol}: {str(e)}")
            if fallback:
                # Try with compact data instead of full
                if outputsize == "full":
                    logger.info(f"Trying compact data for {symbol} instead of full")
                    return self.get_daily_adjusted(symbol, outputsize="compact", cache=cache, fallback=False)
                # Try non-adjusted data
                try:
                    logger.info(f"Trying non-adjusted daily data for {symbol}")
                    return self.get_daily(symbol, outputsize=outputsize, cache=cache, fallback=False)
                except Exception:
                    raise NotEnoughDataError(f"No daily data available for {symbol}")
            else:
                raise
                
        except Exception as e:
            logger.error(f"Error fetching daily adjusted data for {symbol}: {str(e)}")
            if fallback:
                try:
                    logger.info(f"Trying non-adjusted daily data for {symbol}")
                    return self.get_daily(symbol, outputsize=outputsize, cache=cache, fallback=False)
                except Exception:
                    raise AlphaVantageError(f"Failed to fetch daily data for {symbol}: {str(e)}")
            else:
                raise
    
    def get_daily(self, symbol, outputsize="full", cache=True, cache_days=1, fallback=False):
        """
        Get daily time series for a symbol (non-adjusted).
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        outputsize : str, optional
            'compact' for latest 100 data points, 'full' for up to 20 years of data
        cache : bool, optional
            Whether to use cached data
        cache_days : int, optional
            Number of days before cache expires
        fallback : bool, optional
            Whether to try alternative approaches if the request fails
            
        Returns:
        --------
        pd.DataFrame
            Daily time series data
        """
        # Define cache file path
        cache_file = f"{self.cache_dir}/{symbol}_daily.csv"
        
        # Check if cache file exists and is recent
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < cache_days * 86400:
                logger.info(f"Using cached daily data for {symbol}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Prepare API parameters
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'datatype': 'json'
        }
        
        try:
            # Make the request
            data = self._make_request(params, use_cache=cache)
            
            # Parse the time series data
            time_series_key = 'Time Series (Daily)'
            
            if time_series_key not in data:
                raise NotEnoughDataError(f"No time series data found for {symbol}")
            
            time_series = data.get(time_series_key, {})
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns and convert to numeric
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.apply(pd.to_numeric)
            
            # Add adjusted_close (same as close for non-adjusted data)
            df['adjusted_close'] = df['close']
            
            # Calculate returns
            df['returns'] = df['adjusted_close'].pct_change()
            
            # Cache the result if requested
            if cache:
                df.to_csv(cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            if fallback:
                try:
                    # Try weekly data if daily unavailable
                    logger.info(f"Trying weekly data for {symbol}")
                    weekly_df = self.get_weekly(symbol, cache=cache, fallback=False)
                    
                    # Resample to daily
                    logger.info(f"Resampling weekly data to daily for {symbol}")
                    weekly_df = weekly_df.asfreq('D', method='ffill')
                    
                    return weekly_df
                except Exception:
                    raise AlphaVantageError(f"Failed to fetch daily or weekly data for {symbol}")
            else:
                raise
    
    

    def get_technical_indicator(self, symbol, indicator, time_period=20, series_type='close', cache=True, cache_days=1):
        """
        Get technical indicator for a symbol with enhanced error handling.
        Adds special handling for commodity symbols.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        indicator : str
            Technical indicator function name (e.g., 'SMA', 'EMA', 'RSI')
        time_period : int, optional
            Time period for the indicator
        series_type : str, optional
            Price series to use (e.g., 'close', 'open', 'high', 'low')
        cache : bool, optional
            Whether to use cached data
        cache_days : int, optional
            Number of days before cache expires
                
        Returns:
        --------
        pd.DataFrame
            Technical indicator data
        """
        # Define commodity symbols that need special handling
        commodity_symbols = ['BRENT', 'WTI', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE']
        
        # Check if symbol is a commodity
        if symbol in commodity_symbols:
            logger.info(f"Using commodity-specific calculation for {indicator} on {symbol}")
            return self.get_commodity_technical_indicator(symbol, indicator, time_period, series_type, cache)
        
        # for non-commodity symbols:
        # Define cache file path
        cache_file = f"{self.cache_dir}/{symbol}_{indicator}_{time_period}.csv"
        
        # Check if cache file exists and is recent
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < cache_days * 86400:
                logger.info(f"Using cached {indicator} data for {symbol}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Prepare API parameters
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': 'daily',
            'time_period': time_period,
            'series_type': series_type,
            'datatype': 'json'
        }
        
        try:
            # Make the request
            data = self._make_request(params, use_cache=cache)
            
            # Parse the indicator data
            indicator_key = f"Technical Analysis: {indicator}"
            indicator_data = data.get(indicator_key, {})
            
            if not indicator_data:
                raise NotEnoughDataError(f"No {indicator} data found for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(indicator_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert to numeric
            df = df.apply(pd.to_numeric)
            
            # Cache the result if requested
            if cache:
                df.to_csv(cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {symbol}: {str(e)}")
            
            # Try to calculate the indicator locally as fallback
            try:
                logger.info(f"Calculating {indicator} locally for {symbol}")
                return self._calculate_indicator_locally(symbol, indicator, time_period, series_type)
            except Exception as local_e:
                logger.error(f"Local calculation failed: {str(local_e)}")
                raise AlphaVantageError(f"Failed to get {indicator} for {symbol}: {str(e)}")
                
   


    def get_commodity_technical_indicator(self, commodity_symbol, indicator_name, time_period=20, series_type='close', cache=True):
        """
        Calculate technical indicators for commodities using locally computed methods
        since Alpha Vantage doesn't support technical indicator endpoints for commodities.
        
        Parameters:
        -----------
        commodity_symbol : str
            Commodity symbol (BRENT, WTI, etc.)
        indicator_name : str
            Technical indicator to calculate (SMA, EMA, RSI, etc.)
        time_period : int
            Period for indicator calculation
        series_type : str
            Price series to use (close, open, etc.)
        cache : bool
            Whether to use cached data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated indicator
        """
        try:
            # First get the commodity data
            from . import CommodityData
            if not hasattr(self, '_commodity_client'):
                self._commodity_client = CommodityData(self)
            
            # Get commodity price data
            commodity_data = self._commodity_client.get_commodity_data(
                commodity=commodity_symbol,
                interval='monthly',
                cache=cache
            )
            
            # If no data returned, raise error
            if commodity_data is None or commodity_data.empty:
                raise ValueError(f"No commodity data found for {commodity_symbol}")
            
            # Extract price series to use for indicator calculation
            if 'value' in commodity_data.columns:
                price_series = commodity_data['value']
            elif 'close' in commodity_data.columns:
                price_series = commodity_data['close']
            else:
                # Try to find any numeric column
                numeric_cols = commodity_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_series = commodity_data[numeric_cols[0]]
                else:
                    raise ValueError(f"No suitable price column found for {commodity_symbol}")
            
            # Calculate the indicator
            if indicator_name == 'SMA':
                result = price_series.rolling(window=time_period).mean()
                return pd.DataFrame({f'SMA': result})
                
            elif indicator_name == 'EMA':
                result = price_series.ewm(span=time_period, adjust=False).mean()
                return pd.DataFrame({f'EMA': result})
                
            elif indicator_name == 'RSI':
                # Calculate RSI
                delta = price_series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=time_period).mean()
                avg_loss = loss.rolling(window=time_period).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                return pd.DataFrame({f'RSI': rsi})
                
            elif indicator_name == 'MACD':
                # Calculate MACD with standard parameters
                ema12 = price_series.ewm(span=12, adjust=False).mean()
                ema26 = price_series.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                
                return pd.DataFrame({
                    'MACD': macd_line,
                    'MACD_Signal': signal_line,
                    'MACD_Hist': macd_line - signal_line
                })
            
            else:
                raise ValueError(f"Unsupported indicator {indicator_name} for commodities")
        
        except Exception as e:
            logger.error(f"Error calculating {indicator_name} for {commodity_symbol}: {str(e)}")
            # Return empty dataframe
            return pd.DataFrame()
    
    def _calculate_indicator_locally(self, symbol, indicator, time_period, series_type):
        """
        Calculate technical indicator locally when API fails.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        indicator : str
            Technical indicator function name
        time_period : int
            Time period for the indicator
        series_type : str
            Price series to use
            
        Returns:
        --------
        pd.DataFrame
            Technical indicator data
        """
        # Get price data
        try:
            df = self.get_daily_adjusted(symbol, outputsize="full")
        except Exception:
            df = self.get_daily(symbol, outputsize="full")
        
        # Choose price series
        if series_type == 'close':
            price_series = df['close']
        elif series_type == 'open':
            price_series = df['open']
        elif series_type == 'high':
            price_series = df['high']
        elif series_type == 'low':
            price_series = df['low']
        else:
            price_series = df['close']
        
        # Calculate different indicators
        if indicator == 'SMA':
            result = price_series.rolling(window=time_period).mean()
            return pd.DataFrame({indicator: result})
            
        elif indicator == 'EMA':
            result = price_series.ewm(span=time_period, adjust=False).mean()
            return pd.DataFrame({indicator: result})
            
        elif indicator == 'RSI':
            # Calculate RSI
            delta = price_series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=time_period).mean()
            avg_loss = loss.rolling(window=time_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return pd.DataFrame({indicator: rsi})
            
        elif indicator == 'MACD':
            # Calculate MACD
            ema12 = price_series.ewm(span=12, adjust=False).mean()
            ema26 = price_series.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            return pd.DataFrame({
                'MACD': macd_line,
                'MACD_Signal': signal_line,
                'MACD_Hist': macd_line - signal_line
            })
        
        else:
            raise ValueError(f"Local calculation not supported for indicator: {indicator}")
    
    def get_economic_indicator(self, indicator, cache=True, cache_days=1):
        """
        Get economic indicator data with enhanced validation.
        
        Parameters:
        -----------
        indicator : str
            Economic indicator name (e.g., 'REAL_GDP', 'CPI')
        cache : bool, optional
            Whether to use cached data
        cache_days : int, optional
            Number of days before cache expires
            
        Returns:
        --------
        pd.DataFrame
            Economic indicator data
        """
        # Define cache file path
        cache_file = f"{self.cache_dir}/economic_{indicator}.csv"
        
        # Check if cache file exists and is recent
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < cache_days * 86400:
                logger.info(f"Using cached economic data for {indicator}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Prepare API parameters
        params = {
            'function': indicator,
            'datatype': 'json'
        }
        
        try:
            # Make the request
            data = self._make_request(params, use_cache=cache)
            
            # Parse the indicator data
            indicator_data = data.get('data', [])
            
            if not indicator_data:
                raise AlphaVantageError(f"No data found for economic indicator {indicator}")
            
            # Convert to DataFrame
            df = pd.DataFrame(indicator_data)
            
            # Set date as index
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Cache the result if requested
            if cache:
                df.to_csv(cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching economic indicator {indicator}: {str(e)}")
            raise
    
    def get_news_sentiment(self, tickers=None, topics=None, time_from=None, cache=True, cache_days=1):
        """
        Get news sentiment for tickers or topics with better error handling.
        
        Parameters:
        -----------
        tickers : str or list, optional
            Ticker symbols to get sentiment for
        topics : str or list, optional
            Topics to get sentiment for (e.g., 'earnings', 'mergers_and_acquisitions')
        time_from : str, optional
            Time to get sentiment from (e.g., '20220410T0130') - defaults to 7 days ago
        cache : bool, optional
            Whether to use cached data if available
        cache_days : int, optional
            Number of days before cache expires
                
        Returns:
        --------
        pd.DataFrame
            News sentiment data
        """
        # Format tickers or topics
        if tickers:
            if isinstance(tickers, list):
                tickers = ','.join(tickers)
            param_name = 'tickers'
            param_value = tickers
            cache_key = f"sentiment_tickers_{tickers.replace(',', '_')}"
        elif topics:
            if isinstance(topics, list):
                topics = ','.join(topics)
            param_name = 'topics'
            param_value = topics
            cache_key = f"sentiment_topics_{topics.replace(',', '_')}"
        else:
            # Default to economy-related topics if nothing specified
            param_name = 'topics'
            param_value = 'economy_fiscal,economy_monetary,economy_macro'
            cache_key = "sentiment_economy"
        
        cache_file = f"{self.cache_dir}/{cache_key}.csv"
        
        # Check if cache file exists and is recent
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < cache_days * 86400:
                logger.info(f"Using cached sentiment data for {param_value}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Default time from if not specified (format as required by the API)
        if not time_from:
            # Create a timestamp 7 days ago in YYYYMMDDTHHMM format
            seven_days_ago = datetime.now() - timedelta(days=7)
            time_from = seven_days_ago.strftime('%Y%m%dT%H%M')
        
        # Prepare API parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            param_name: param_value,
            'time_from': time_from,
            'limit': 200,  # Maximum allowed
            'sort': 'RELEVANCE'
        }
        
        try:
            # Make the request
            data = self._make_request(params, use_cache=cache)
            
            # Check if we received valid data
            if not isinstance(data, dict) or 'feed' not in data:
                logger.warning(f"Invalid or empty response from NEWS_SENTIMENT API")
                return pd.DataFrame()
            
            # Parse the sentiment data
            feed = data.get('feed', [])
            
            if not feed:
                logger.warning(f"No sentiment data found for {param_name}={param_value}")
                return pd.DataFrame()
            
            # Extract relevant data
            sentiment_data = []
            
            for article in feed:
                # Get the publish time
                try:
                    publish_time = datetime.strptime(article.get('time_published', ''), '%Y%m%dT%H%M%S')
                except:
                    # Use current time if cannot parse
                    publish_time = datetime.now()
                
                # Get overall sentiment
                overall_sentiment_score = article.get('overall_sentiment_score', 0)
                
                # Get ticker-specific sentiment
                ticker_sentiment = article.get('ticker_sentiment', [])
                
                # Process ticker sentiment
                for ticker_data in ticker_sentiment:
                    ticker = ticker_data.get('ticker', '')
                    relevance_score = float(ticker_data.get('relevance_score', 0))
                    ticker_sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0))
                    
                    sentiment_data.append({
                        'date': publish_time,
                        'ticker': ticker,
                        'relevance_score': relevance_score,
                        'ticker_sentiment_score': ticker_sentiment_score,
                        'overall_sentiment_score': float(overall_sentiment_score),
                        'title': article.get('title', ''),
                        'url': article.get('url', '')
                    })
            
            # Convert to DataFrame
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                df.set_index('date', inplace=True)
                df = df.sort_index()
                
                # Cache the result if requested
                if cache:
                    df.to_csv(cache_file)
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {str(e)}")
            return pd.DataFrame()
    
    def get_fx_rate(self, from_currency, to_currency='USD', interval='daily', cache=True, cache_days=1):
        """
        Get foreign exchange rate data with flexible time intervals and better error handling.
        
        Parameters:
        -----------
        from_currency : str
            From currency code (e.g., 'EUR')
        to_currency : str, optional
            To currency code (default: 'USD')
        interval : str, optional
            Time interval - 'daily', 'weekly', or 'monthly' (default: 'daily')
        cache : bool, optional
            Whether to use cached data
        cache_days : int, optional
            Number of days before cache expires
                
        Returns:
        --------
        pd.DataFrame
            Foreign exchange rate data
        """
        # Handle currency pairs in format "EUR/USD"
        if '/' in from_currency:
            parts = from_currency.split('/')
            if len(parts) == 2:
                from_currency, to_currency = parts
        
        # Create a more specific cache file name including the interval
        cache_file = f"{self.cache_dir}/fx_{from_currency}_{to_currency}_{interval}.csv"
        
        # Check if cache file exists and is recent
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < cache_days * 86400:
                logger.info(f"Using cached FX data for {from_currency}/{to_currency}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Map interval to API function
        interval_map = {
            'daily': 'FX_DAILY',
            'weekly': 'FX_WEEKLY',
            'monthly': 'FX_MONTHLY'
        }
        
        function = interval_map.get(interval.lower(), 'FX_DAILY')
        
        # Prepare API parameters
        params = {
            'function': function,
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'datatype': 'json'
        }
        
        try:
            # Make the request
            data = self._make_request(params, use_cache=cache)
            
            # Parse the FX data - note the different key format for different intervals
            key_prefix = {
                'FX_DAILY': 'Time Series FX (Daily)',
                'FX_WEEKLY': 'Time Series FX (Weekly)',
                'FX_MONTHLY': 'Time Series FX (Monthly)'
            }
            
            time_series_key = key_prefix.get(function)
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                raise NotEnoughDataError(f"No FX data found for {from_currency}/{to_currency}")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns and convert to numeric
            df.columns = ['open', 'high', 'low', 'close']
            df = df.apply(pd.to_numeric)
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Cache the result if requested
            if cache:
                df.to_csv(cache_file)
            
            return df
                
        except NotEnoughDataError:
            logger.warning(f"No {interval} FX data found for {from_currency}/{to_currency}")
            
            # Try again with a different interval if no data found
            if interval == 'daily':
                logger.info(f"Trying weekly data for {from_currency}/{to_currency}")
                return self.get_fx_rate(from_currency, to_currency, 'weekly', cache, cache_days)
            elif interval == 'weekly':
                logger.info(f"Trying monthly data for {from_currency}/{to_currency}")
                return self.get_fx_rate(from_currency, to_currency, 'monthly', cache, cache_days)
            else:
                raise NotEnoughDataError(f"No FX data available for {from_currency}/{to_currency}")
                    
        except Exception as e:
            logger.error(f"Error fetching FX data for {from_currency}/{to_currency}: {str(e)}")
            
            # Try again with a different interval if there was an error
            if interval == 'daily':
                logger.info(f"Trying weekly data instead for {from_currency}/{to_currency}")
                return self.get_fx_rate(from_currency, to_currency, 'weekly', cache, cache_days)
            elif interval == 'weekly':
                logger.info(f"Trying monthly data instead for {from_currency}/{to_currency}")
                return self.get_fx_rate(from_currency, to_currency, 'monthly', cache, cache_days)
            else:
                # If we've already tried monthly or all else fails, re-raise the exception
                raise AlphaVantageError(f"Failed to fetch FX data for {from_currency}/{to_currency}: {str(e)}")

class CommodityData:
    """
    Specialized class for handling commodity data from Alpha Vantage with proper
    fallbacks, normalizations, and reliable data retrieval.
    """
    
    # Mapping of commodity names to proper symbols
    COMMODITY_SYMBOLS = {
        'wti': 'WTI',
        'wti_oil': 'WTI',
        'crude_oil': 'WTI',
        'brent': 'BRENT',
        'brent_oil': 'BRENT',
        'natural_gas': 'NATURAL_GAS',
        'natgas': 'NATURAL_GAS',
        'copper': 'COPPER',
        'aluminum': 'ALUMINUM',
        'wheat': 'WHEAT',
        'corn': 'CORN',
        'cotton': 'COTTON',
        'sugar': 'SUGAR',
        'coffee': 'COFFEE',
        'all': 'ALL_COMMODITIES'
    }
    
    def __init__(self, av_client):
        """
        Initialize the CommodityData manager.
        
        Parameters:
        -----------
        av_client : AlphaVantageClient
            Alpha Vantage client instance
        """
        self.av_client = av_client
        self.logger = logging.getLogger(__name__)
    
    def get_commodity_data(self, commodity, interval="monthly", cache=True, fallback=True):
        """
        Get commodity data with proper error handling and fallbacks.
        
        Parameters:
        -----------
        commodity : str
            Commodity name or symbol
        interval : str, optional
            Time interval ('daily', 'weekly', 'monthly')
        cache : bool, optional
            Whether to use cached data
        fallback : bool, optional
            Whether to try alternative approaches if the request fails
            
        Returns:
        --------
        pd.DataFrame
            Normalized commodity data
        """
        # Normalize commodity name
        symbol = self.COMMODITY_SYMBOLS.get(commodity.lower(), commodity.upper())
        
        # Create cache file path
        cache_file = f"{self.av_client.cache_dir}/commodity_{symbol}_{interval}.csv"
        
        # Check if cache file exists and is valid
        if cache and os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            # Commodities don't update that frequently, so we can use a longer cache TTL
            # 7 days for monthly, 3 days for weekly, 1 day for daily
            ttl_days = 7 if interval == 'monthly' else 3 if interval == 'weekly' else 1
            if (time.time() - file_time) < ttl_days * 86400:
                self.logger.info(f"Using cached {interval} data for commodity {symbol}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Prepare API parameters - commodity endpoints use function as the symbol
        params = {
            'function': symbol,
            'interval': interval,
            'datatype': 'json'
        }
        
        try:
            # Make the request
            data = self.av_client._make_request(params, use_cache=cache)
            
            # Process the data
            if not data or not isinstance(data, dict) or 'data' not in data:
                raise NotEnoughDataError(f"No data returned for commodity {symbol}")
            
            commodity_data = data.get('data', [])
            
            if not commodity_data or len(commodity_data) == 0:
                raise NotEnoughDataError(f"Empty data list for commodity {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(commodity_data)
            
            # Set date as index
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            
            # Convert value to numeric
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Calculate returns if possible
            if 'value' in df.columns:
                df['returns'] = df['value'].pct_change()
            
            # Apply commodity-specific normalization
            df = self._normalize_data(df, symbol)
            
            # Cache the result if requested
            if cache:
                df.to_csv(cache_file)

            if isinstance(df, pd.DataFrame) and not df.empty:
            # Ensure we have a 'value' column for consistency
                if 'value' not in df.columns:
                    if 'close' in df.columns:
                        # Copy close to value for consistency
                        df['value'] = df['close']
                    else:
                        # Find any numeric column to use as value
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            df['value'] = df[numeric_cols[0]]
                
                # Add returns column if not present
                if 'returns' not in df.columns and 'value' in df.columns:
                    df['returns'] = df['value'].pct_change()
                
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex) and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    df = df.sort_index()
            
            return df
            
        except NotEnoughDataError as e:
            self.logger.warning(f"Not enough data for commodity {symbol}: {str(e)}")
            
            if fallback:
                # Try with a different interval
                if interval == 'daily':
                    self.logger.info(f"Trying weekly data for commodity {symbol}")
                    return self.get_commodity_data(symbol, interval='weekly', cache=cache, fallback=True)
                elif interval == 'weekly':
                    self.logger.info(f"Trying monthly data for commodity {symbol}")
                    return self.get_commodity_data(symbol, interval='monthly', cache=cache, fallback=True)
                else:
                    # Try alternative commodity source if all else fails
                    return self._get_from_alternative_source(symbol, interval, cache)
            else:
                raise
                
        except Exception as e:
            self.logger.error(f"Error getting data for commodity {symbol}: {str(e)}")
            
            if fallback:
                # Try with a different interval
                if interval == 'daily':
                    self.logger.info(f"Trying weekly data for commodity {symbol}")
                    return self.get_commodity_data(symbol, interval='weekly', cache=cache, fallback=True)
                elif interval == 'weekly':
                    self.logger.info(f"Trying monthly data for commodity {symbol}")
                    return self.get_commodity_data(symbol, interval='monthly', cache=cache, fallback=True)
                else:
                    # Try alternative commodity source if all else fails
                    return self._get_from_alternative_source(symbol, interval, cache)
            else:
                raise
    
    def _normalize_data(self, df, symbol):
        """
        Normalize commodity data based on type.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw commodity data
        symbol : str
            Commodity symbol
            
        Returns:
        --------
        pd.DataFrame
            Normalized data
        """
        # Ensure consistent column names
        if 'value' in df.columns:
            # Rename for clarity based on commodity type
            commodity_type = self._get_commodity_type(symbol)
            
            if commodity_type == 'energy':
                # Energy commodities are typically priced in USD per barrel/unit
                df.rename(columns={'value': 'price_usd'}, inplace=True)
            elif commodity_type == 'metal':
                # Metals typically priced in USD per unit
                df.rename(columns={'value': 'price_usd'}, inplace=True)
            elif commodity_type == 'agriculture':
                # Agricultural commodities typically priced per bushel/contract
                df.rename(columns={'value': 'price_usd'}, inplace=True)
            else:
                # Generic price column
                df.rename(columns={'value': 'price'}, inplace=True)
        
        # Add standard close column for consistency with other financial data
        price_col = next((col for col in df.columns if col.startswith('price') or col == 'value'), None)
        if price_col:
            df['close'] = df[price_col]
            
            # Add other standard columns for compatibility
            df['open'] = df[price_col]
            df['high'] = df[price_col]
            df['low'] = df[price_col]
        
        return df
    
    def _get_commodity_type(self, symbol):
        """
        Determine commodity type for appropriate normalization.
        
        Parameters:
        -----------
        symbol : str
            Commodity symbol
            
        Returns:
        --------
        str
            Commodity type ('energy', 'metal', 'agriculture', or 'other')
        """
        symbol = symbol.lower()
        
        if symbol in ['wti', 'brent', 'natural_gas']:
            return 'energy'
        elif symbol in ['copper', 'aluminum', 'gold', 'silver']:
            return 'metal'
        elif symbol in ['wheat', 'corn', 'cotton', 'sugar', 'coffee']:
            return 'agriculture'
        else:
            return 'other'
    
    def _get_from_alternative_source(self, symbol, interval, cache=True):
        """
        Get commodity data from alternative source when Alpha Vantage fails.
        Currently tries using time series API endpoints as fallback.
        
        Parameters:
        -----------
        symbol : str
            Commodity symbol
        interval : str
            Time interval
        cache : bool, optional
            Whether to use cached data
            
        Returns:
        --------
        pd.DataFrame
            Commodity data from alternative source
        """
        self.logger.info(f"Trying to get commodity {symbol} from alternative source")
        
        # Try using time series API endpoints if commodity endpoints fail
        # For example, some commodity ETFs or proxies
        commodity_etf_map = {
            'WTI': 'USO',      # United States Oil Fund
            'BRENT': 'BNO',    # United States Brent Oil Fund
            'NATURAL_GAS': 'UNG',  # United States Natural Gas Fund
            'COPPER': 'CPER',  # United States Copper Index Fund
            'WHEAT': 'WEAT',   # Teucrium Wheat Fund
            'CORN': 'CORN',    # Teucrium Corn Fund
        }
        
        etf_symbol = commodity_etf_map.get(symbol.upper())
        
        if etf_symbol:
            self.logger.info(f"Using ETF {etf_symbol} as proxy for commodity {symbol}")
            
            try:
                # Get ETF data using time series API
                if interval == 'daily':
                    etf_data = self.av_client.get_daily_adjusted(etf_symbol, cache=cache)
                elif interval == 'weekly':
                    etf_data = self.av_client.get_weekly(etf_symbol, cache=cache)
                elif interval == 'monthly':
                    etf_data = self.av_client.get_monthly(etf_symbol, cache=cache)
                
                # Normalize to match commodity data format
                if not etf_data.empty:
                    # Create a new DataFrame with commodity format
                    commodity_df = pd.DataFrame(index=etf_data.index)
                    commodity_df['value'] = etf_data['adjusted_close']
                    commodity_df['returns'] = etf_data['returns']
                    
                    # Apply normalization
                    commodity_df = self._normalize_data(commodity_df, symbol)
                    
                    self.logger.info(f"Successfully created proxy data for {symbol} using {etf_symbol}")
                    return commodity_df
            except Exception as e:
                self.logger.error(f"Error creating proxy data for {symbol}: {str(e)}")
        
        # If all else fails, return empty DataFrame
        self.logger.warning(f"All attempts to get commodity {symbol} data failed")
        return pd.DataFrame()




def get_all_asset_returns(av_client, assets_dict, lookback_days=1460, cache=True):
    """
    Enhanced function to get returns data for all assets with longer history and better fallbacks.
    
    Parameters:
    -----------
    av_client : AlphaVantageClient
        Alpha Vantage client instance
    assets_dict : dict
        Dictionary of asset classes and tickers
    lookback_days : int, optional
        Number of days to look back for returns (increased from 730 to 1460)
    cache : bool, optional
        Whether to use cached data
    
    Returns:
    --------
    pd.DataFrame
        Returns data for all assets
    """
    logger.info(f"Loading returns data with extended {lookback_days}-day history...")
    
    # Create commodity data handler
    commodity_data = CommodityData(av_client)
    
    all_returns = pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # Try to load from cache first with updated timestamp check
    cache_file = "cached_returns.csv"
    if cache and os.path.exists(cache_file):
        file_time = os.path.getmtime(cache_file)
        hours_old = (time.time() - file_time) / 3600
        
        # Use cached data if less than 12 hours old (reduced from 24)
        if hours_old < 12:
            try:
                cached_returns = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Using cached returns data ({hours_old:.1f} hours old)")
                
                # Verify data adequacy - need at least 60 data points per asset
                inadequate_assets = []
                for asset_class, tickers in assets_dict.items():
                    for ticker in tickers:
                        ticker_norm = ticker.replace('/', '_') if '/' in ticker else ticker
                        if ticker_norm not in cached_returns.columns or cached_returns[ticker_norm].count() < 60:
                            inadequate_assets.append(ticker)
                
                if not inadequate_assets:
                    logger.info("All assets have adequate data in cache")
                    return cached_returns
                else:
                    logger.info(f"Need to fetch additional data for {len(inadequate_assets)} assets")
            except Exception as e:
                logger.warning(f"Error loading cached returns: {str(e)}")
    
    # Process each asset class with better error handling and proxy mapping
    proxy_mapping = {
        # Commodity proxies (ETFs with longer history)
        'BRENT': 'BNO',  # Brent Oil ETF
        'WTI': 'USO',    # WTI Oil ETF
        'NATURAL_GAS': 'UNG',  # Natural Gas ETF
        'COPPER': 'CPER', # Copper ETF
        'ALUMINUM': 'JJU', # Aluminum ETF
        'WHEAT': 'WEAT',  # Wheat ETF
        'CORN': 'CORN',   # Corn ETF
        
        # Bond proxies
        'GOVT': 'IEF',    # Use IEF as proxy for GOVT if needed
        
        # Equity proxies (rarely needed but included for completeness)
        'VGK': 'EZU'      # European stocks alternative
    }
    
    # Track fetched assets to avoid duplicates
    fetched_assets = set()
    
    # First, process each asset using direct data
    for asset_class, tickers in assets_dict.items():
        logger.info(f"Processing {asset_class}...")
        
        for ticker in tickers:
            if ticker in fetched_assets:
                continue
                
            try:
                # Try to get data directly for the ticker
                if asset_class == 'Commodities':
                    df = commodity_data.get_commodity_data(ticker, interval='daily', cache=cache)
                elif asset_class == 'Currencies':
                    from_currency, to_currency = _parse_currency_pair(ticker)
                    df = av_client.get_fx_rate(from_currency, to_currency, interval='daily', cache=cache)
                else:
                    df = av_client.get_daily_adjusted(ticker, cache=cache)
                
                # Check if we have enough data
                returns_col = _find_returns_column(df)
                
                if returns_col is not None and len(df[returns_col].dropna()) >= 60:
                    all_returns[ticker] = df[returns_col]
                    fetched_assets.add(ticker)
                    logger.info(f"Added direct returns for {ticker} ({len(df[returns_col].dropna())} points)")
                else:
                    # Not enough direct data, will try proxy in next step
                    logger.warning(f"Insufficient direct data for {ticker}")
            except Exception as e:
                logger.warning(f"Error getting direct data for {ticker}: {str(e)}")
    
    # Next, try proxy assets for those missing adequate data
    for asset_class, tickers in assets_dict.items():
        for ticker in tickers:
            if ticker in fetched_assets:
                continue
                
            # Try to use proxy if available
            proxy = proxy_mapping.get(ticker)
            if proxy:
                try:
                    logger.info(f"Trying proxy {proxy} for {ticker}")
                    df_proxy = av_client.get_daily_adjusted(proxy, cache=cache)
                    
                    if 'returns' in df_proxy.columns and len(df_proxy['returns'].dropna()) >= 60:
                        # Use proxy data and rename the column
                        all_returns[ticker] = df_proxy['returns']
                        fetched_assets.add(ticker)
                        logger.info(f"Added proxy returns for {ticker} using {proxy} ({len(df_proxy['returns'].dropna())} points)")
                    else:
                        logger.warning(f"Proxy {proxy} for {ticker} also has insufficient data")
                except Exception as e:
                    logger.warning(f"Error getting proxy data for {ticker}: {str(e)}")
    
    # After fetching all available data, align dates and handle missing values
    if not all_returns.empty:
        # Ensure all returns start from the same date, if possible
        try:
            all_returns = all_returns[all_returns.index >= start_date]
        except:
            pass
            
        # Fill missing values using forward and backward fill
        all_returns = all_returns.fillna(method='ffill').fillna(method='bfill')
        
        # Cache the improved data
        all_returns.to_csv(cache_file)
        logger.info(f"Saved enhanced returns data with {len(all_returns.columns)} assets")
    
    return all_returns

# Helper functions for the refactored get_all_asset_returns
def _parse_currency_pair(ticker):
    """Parse currency ticker into components"""
    if '/' in ticker:
        from_currency, to_currency = ticker.split('/')
        return from_currency, to_currency
    elif ticker in ['JPY', 'CAD', 'CHF']:
        return 'USD', ticker
    else:
        return ticker, 'USD'

def _find_returns_column(df):
    """Find the returns column in a dataframe"""
    if df is None or df.empty:
        return None
        
    if 'returns' in df.columns:
        return 'returns'
    elif df.shape[0] > 1:
        # Try to find a suitable price column to calculate returns
        for col in ['close', 'adjusted_close', 'value', 'price']:
            if col in df.columns:
                # Add returns if not present
                df['returns'] = df[col].pct_change()
                return 'returns'
    
    # Cannot find or calculate returns
    return None