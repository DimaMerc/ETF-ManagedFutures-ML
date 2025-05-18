# error_handlers_and_utils.py

import os
import time
import logging
import functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions for Alpha Vantage API
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

def classify_alpha_vantage_error(response):
    """
    Classify Alpha Vantage API errors based on response.
    
    Parameters:
    -----------
    response : dict
        API response data
        
    Returns:
    --------
    tuple
        (error_type, error_message)
    """
    if isinstance(response, dict):
        # Check for explicit error messages
        if "Error Message" in response:
            error_msg = response["Error Message"]
            if "Invalid API call" in error_msg:
                return "INVALID_REQUEST", error_msg
            elif "not found" in error_msg.lower():
                return "SYMBOL_NOT_FOUND", error_msg
            else:
                return "API_ERROR", error_msg
        
        # Check for rate limiting
        if "Note" in response and "call frequency" in response["Note"]:
            return "RATE_LIMIT", response["Note"]
            
        # Check for valid structure but empty data
        # Time series data
        time_series_key = next((k for k in response.keys() 
                              if k.startswith("Time Series") or 
                              k.startswith("Monthly") or
                              k.startswith("Weekly")),
                             None)
        
        if time_series_key and time_series_key != "Meta Data":
            time_series_data = response.get(time_series_key, {})
            if len(time_series_data) == 0:
                return "NO_DATA", "No data points available"
            
        # Check for commodity data structure but empty data
        if "data" in response and isinstance(response["data"], list):
            if len(response["data"]) == 0:
                return "NO_DATA", "No commodity data points available"
    
    # Check for truly empty response
    if not response or (isinstance(response, dict) and len(response) <= 1):
        return "EMPTY_RESPONSE", "Empty or minimal response received"
    
    # If we get here, response appears valid
    return "VALID", "Response appears valid"

def handle_not_enough_data(symbol, function, client, interval="daily"):
    """
    Handle the 'not enough data' error case gracefully with multiple fallbacks.
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol
    function : str
        API function to call
    client : AlphaVantageClient
        Alpha Vantage client instance
    interval : str, optional
        Time interval (daily, weekly, monthly)
        
    Returns:
    --------
    dict or pd.DataFrame
        API response or processed data
    """
    logger.info(f"Handling potential 'not enough data' for {symbol} with {function}")
    
    # Try strategies based on function type
    if function.startswith('TIME_SERIES'):
        return _handle_time_series_data_shortage(symbol, function, client, interval)
    elif function in ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM']:
        return _handle_commodity_data_shortage(symbol, client, interval)
    elif function.startswith('FX_'):
        return _handle_forex_data_shortage(symbol, function, client, interval)
    else:
        # For other API functions, try compact data
        try:
            # Try with compact output size first
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': 'compact',
                'interval': interval,
                'datatype': 'json'
            }
            response = client._make_request(params)
            is_valid, _ = classify_alpha_vantage_error(response)
            
            if is_valid == "VALID":
                logger.info(f"Successfully retrieved compact data for {symbol}")
                return response
            else:
                logger.warning(f"Compact data retrieval failed for {symbol}")
                raise NotEnoughDataError(f"No data available for {symbol} with function {function}")
                
        except Exception as e:
            logger.error(f"All data retrieval attempts failed for {symbol}: {str(e)}")
            raise NotEnoughDataError(f"All data retrieval attempts failed for {symbol}")

def _handle_time_series_data_shortage(symbol, function, client, interval):
    """
    Handle data shortage for time series functions.
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol
    function : str
        Time series function
    client : AlphaVantageClient
        Alpha Vantage client instance
    interval : str
        Time interval
        
    Returns:
    --------
    dict
        API response
    """
    fallback_chain = []
    
    # Build fallback chain based on function and interval
    if function == 'TIME_SERIES_DAILY_ADJUSTED':
        fallback_chain = [
            # Try compact data first
            {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'outputsize': 'compact'},
            # Then try non-adjusted daily
            {'function': 'TIME_SERIES_DAILY', 'outputsize': 'compact'},
            # Then weekly data
            {'function': 'TIME_SERIES_WEEKLY_ADJUSTED'},
            # Finally monthly data
            {'function': 'TIME_SERIES_MONTHLY_ADJUSTED'}
        ]
    elif function == 'TIME_SERIES_DAILY':
        fallback_chain = [
            # Try compact data first
            {'function': 'TIME_SERIES_DAILY', 'outputsize': 'compact'},
            # Then try adjusted daily
            {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'outputsize': 'compact'},
            # Then weekly data
            {'function': 'TIME_SERIES_WEEKLY'},
            # Finally monthly data
            {'function': 'TIME_SERIES_MONTHLY'}
        ]
    elif function == 'TIME_SERIES_WEEKLY':
        fallback_chain = [
            # Try adjusted weekly
            {'function': 'TIME_SERIES_WEEKLY_ADJUSTED'},
            # Then monthly data
            {'function': 'TIME_SERIES_MONTHLY'}
        ]
    elif function == 'TIME_SERIES_MONTHLY':
        fallback_chain = [
            # Try adjusted monthly
            {'function': 'TIME_SERIES_MONTHLY_ADJUSTED'},
        ]
    
    # Try each fallback option in sequence
    for i, fallback in enumerate(fallback_chain):
        try:
            logger.info(f"Trying fallback {i+1}/{len(fallback_chain)} for {symbol}: {fallback['function']}")
            
            # Prepare parameters
            params = {
                'symbol': symbol,
                'datatype': 'json',
                **fallback
            }
            
            # Make request
            response = client._make_request(params)
            is_valid, msg = classify_alpha_vantage_error(response)
            
            if is_valid == "VALID":
                logger.info(f"Successfully retrieved data for {symbol} using {fallback['function']}")
                return response
            else:
                logger.warning(f"Fallback {i+1} failed: {msg}")
                
        except Exception as e:
            logger.warning(f"Fallback {i+1} error: {str(e)}")
    
    # If all fallbacks fail, try alternative data source
    logger.warning(f"All fallbacks failed for {symbol}, trying alternative data source")
    return _get_from_alternative_source(symbol, client)

def _handle_commodity_data_shortage(symbol, client, interval):
    """
    Handle data shortage for commodity functions.
    
    Parameters:
    -----------
    symbol : str
        Commodity symbol
    client : AlphaVantageClient
        Alpha Vantage client instance
    interval : str
        Time interval
        
    Returns:
    --------
    dict
        API response
    """
    # Map of intervals to try in sequence
    interval_sequence = {
        'daily': ['weekly', 'monthly'],
        'weekly': ['monthly', 'daily'],
        'monthly': ['weekly', 'daily']
    }
    
    # Try different intervals
    intervals_to_try = interval_sequence.get(interval, ['monthly'])
    
    for alt_interval in intervals_to_try:
        try:
            logger.info(f"Trying {alt_interval} interval for commodity {symbol}")
            
            # Prepare parameters
            params = {
                'function': symbol,
                'interval': alt_interval,
                'datatype': 'json'
            }
            
            # Make request
            response = client._make_request(params)
            is_valid, msg = classify_alpha_vantage_error(response)
            
            if is_valid == "VALID":
                logger.info(f"Successfully retrieved {alt_interval} data for commodity {symbol}")
                return response
            else:
                logger.warning(f"{alt_interval} interval failed: {msg}")
                
        except Exception as e:
            logger.warning(f"{alt_interval} interval error: {str(e)}")
    
    # If all intervals fail, try ETF proxies
    logger.warning(f"All intervals failed for commodity {symbol}, trying ETF proxies")
    
    # Map commodities to ETF proxies
    etf_map = {
        'WTI': 'USO',      # United States Oil Fund
        'BRENT': 'BNO',    # United States Brent Oil Fund
        'NATURAL_GAS': 'UNG',  # United States Natural Gas Fund
        'COPPER': 'CPER',  # United States Copper Index Fund
        'ALUMINUM': 'AA',  # Alcoa (as proxy)
        'WHEAT': 'WEAT',   # Teucrium Wheat Fund
        'CORN': 'CORN',    # Teucrium Corn Fund
        'SUGAR': 'CANE',   # Teucrium Sugar Fund
        'COFFEE': 'JO',    # iPath Bloomberg Coffee Subindex Total Return ETN
        'COTTON': 'BAL'    # iPath Bloomberg Cotton Subindex Total Return ETN
    }
    
    etf_symbol = etf_map.get(symbol.upper())
    
    if etf_symbol:
        try:
            logger.info(f"Using ETF {etf_symbol} as proxy for commodity {symbol}")
            
            # Get ETF data using time series API
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': etf_symbol,
                'outputsize': 'compact',
                'datatype': 'json'
            }
            
            # Make request
            response = client._make_request(params)
            is_valid, msg = classify_alpha_vantage_error(response)
            
            if is_valid == "VALID":
                logger.info(f"Successfully retrieved ETF proxy {etf_symbol} for commodity {symbol}")
                
                # Transform ETF data to match commodity format
                daily_data = response.get('Time Series (Daily)', {})
                
                # Format data to match commodity endpoint
                commodity_data = []
                for date, values in daily_data.items():
                    commodity_data.append({
                        'date': date,
                        'value': float(values.get('4. close', 0))
                    })
                
                # Sort by date (newest first to match API)
                commodity_data.sort(key=lambda x: x['date'], reverse=True)
                
                # Create response matching commodity endpoint format
                formatted_response = {
                    'name': symbol,
                    'interval': interval,
                    'unit': 'USD',
                    'data': commodity_data
                }
                
                return formatted_response
            else:
                logger.warning(f"ETF proxy approach failed: {msg}")
                
        except Exception as e:
            logger.warning(f"ETF proxy error: {str(e)}")
    
    # If all approaches fail
    raise NotEnoughDataError(f"No data available for commodity {symbol}")

def _handle_forex_data_shortage(symbol, function, client, interval):
    """
    Handle data shortage for forex functions.
    
    Parameters:
    -----------
    symbol : str
        Currency pair or symbol
    function : str
        FX function
    client : AlphaVantageClient
        Alpha Vantage client instance
    interval : str
        Time interval
        
    Returns:
    --------
    dict
        API response
    """
    # Parse currency pair
    parts = symbol.split('/')
    from_currency = parts[0] if len(parts) > 0 else symbol
    to_currency = parts[1] if len(parts) > 1 else 'USD'
    
    # Map of intervals to try in sequence
    function_sequence = {
        'FX_DAILY': ['FX_WEEKLY', 'FX_MONTHLY'],
        'FX_WEEKLY': ['FX_MONTHLY', 'FX_DAILY'],
        'FX_MONTHLY': ['FX_WEEKLY', 'FX_DAILY']
    }
    
    # Try different functions
    functions_to_try = function_sequence.get(function, ['FX_DAILY', 'FX_WEEKLY', 'FX_MONTHLY'])
    
    for alt_function in functions_to_try:
        try:
            logger.info(f"Trying {alt_function} for forex {from_currency}/{to_currency}")
            
            # Prepare parameters
            params = {
                'function': alt_function,
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'outputsize': 'compact',
                'datatype': 'json'
            }
            
            # Make request
            response = client._make_request(params)
            is_valid, msg = classify_alpha_vantage_error(response)
            
            if is_valid == "VALID":
                logger.info(f"Successfully retrieved {alt_function} data for forex {from_currency}/{to_currency}")
                return response
            else:
                logger.warning(f"{alt_function} function failed: {msg}")
                
        except Exception as e:
            logger.warning(f"{alt_function} function error: {str(e)}")
    
    # Try inverting the currency pair as last resort
    try:
        logger.info(f"Trying inverted currency pair {to_currency}/{from_currency}")
        
        # Prepare parameters
        params = {
            'function': 'FX_DAILY',
            'from_symbol': to_currency,
            'to_symbol': from_currency,
            'outputsize': 'compact',
            'datatype': 'json'
        }
        
        # Make request
        response = client._make_request(params)
        is_valid, msg = classify_alpha_vantage_error(response)
        
        if is_valid == "VALID":
            logger.info(f"Successfully retrieved inverted pair {to_currency}/{from_currency}")
            
            # Invert the rates
            time_series_key = 'Time Series FX (Daily)'
            if time_series_key in response:
                for date, values in response[time_series_key].items():
                    for key, value in values.items():
                        # Invert each rate
                        values[key] = str(1.0 / float(value))
                
                # Update metadata
                if 'Meta Data' in response:
                    if '2. From Symbol' in response['Meta Data']:
                        response['Meta Data']['2. From Symbol'] = from_currency
                    if '3. To Symbol' in response['Meta Data']:
                        response['Meta Data']['3. To Symbol'] = to_currency
                
                return response
            else:
                logger.warning("Cannot invert rates: time series key not found")
        else:
            logger.warning(f"Inverted pair approach failed: {msg}")
            
    except Exception as e:
        logger.warning(f"Inverted pair error: {str(e)}")
    
    # If all approaches fail
    raise NotEnoughDataError(f"No data available for forex {from_currency}/{to_currency}")

def _get_from_alternative_source(symbol, client):
    """
    Get data from alternative sources when Alpha Vantage fails.
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol
    client : AlphaVantageClient
        Alpha Vantage client instance
        
    Returns:
    --------
    dict or None
        Data from alternative source or None if not available
    """
    # Implement alternative data sources here
    # This could include:
    # - Free alternatives like Yahoo Finance
    # - Local CSV files with historical data
    # - Other paid APIs
    
    logger.warning(f"Alternative data source for {symbol} not implemented")
    return None

# Utility functions for data augmentation (to handle "Not enough data" for LSTM models)
def augment_time_series(data, augmentation_factor=3):
    """
    Augment time series data for better LSTM training.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Original time series data
    augmentation_factor : int
        Number of augmented samples to create
        
    Returns:
    --------
    numpy.ndarray
        Augmented data
    """
    if len(data) == 0:
        return data
        
    augmented_data = [data.copy()]
    
    for i in range(augmentation_factor - 1):
        # 1. Time warping (stretch/compress segments)
        warped = time_warp(data.copy(), warp_factor=np.random.uniform(0.8, 1.2))
        augmented_data.append(warped)
        
        # 2. Magnitude scaling (scale segments while preserving pattern)
        scaled = magnitude_scale(data.copy(), scale_factor=np.random.uniform(0.9, 1.1))
        augmented_data.append(scaled)
        
        # 3. Add subtle noise (market-like fluctuations)
        noised = add_financial_noise(data.copy(), volatility=np.random.uniform(0.001, 0.005))
        augmented_data.append(noised)
    
    return np.concatenate(augmented_data)

def time_warp(data, warp_factor=1.0):
    """
    Apply time warping to a time series.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    warp_factor : float
        Factor to compress or expand time (>1 expands, <1 compresses)
        
    Returns:
    --------
    numpy.ndarray
        Time-warped data
    """
    # Determine new length based on warp factor
    original_length = len(data)
    new_length = int(original_length * warp_factor)
    
    # Ensure minimum length
    new_length = max(new_length, original_length // 2)
    
    # Use linear interpolation to resample
    indices = np.linspace(0, original_length - 1, new_length)
    warped_data = np.interp(indices, np.arange(original_length), data)
    
    # Pad or truncate to original length
    if new_length < original_length:
        # Pad with repeated values
        padding = np.repeat(warped_data[-1], original_length - new_length)
        warped_data = np.concatenate([warped_data, padding])
    elif new_length > original_length:
        # Truncate
        warped_data = warped_data[:original_length]
    
    return warped_data

def magnitude_scale(data, scale_factor=1.0):
    """
    Scale the magnitude of a time series.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    scale_factor : float
        Factor to scale magnitude
        
    Returns:
    --------
    numpy.ndarray
        Magnitude-scaled data
    """
    # Compute mean to preserve
    mean_value = np.mean(data)
    
    # Scale deviations from the mean
    deviations = data - mean_value
    scaled_deviations = deviations * scale_factor
    
    # Reconstruct series with scaled deviations
    scaled_data = mean_value + scaled_deviations
    
    return scaled_data

def add_financial_noise(data, volatility=0.002):
    """
    Add realistic financial noise to a time series.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    volatility : float
        Volatility of the noise
        
    Returns:
    --------
    numpy.ndarray
        Data with added noise
    """
    # Generate random noise based on volatility
    noise = np.random.normal(0, volatility, size=len(data))
    
    # Ensure noise is autocorrelated (more realistic for financial data)
    autocorrelated_noise = np.zeros_like(noise)
    autocorrelated_noise[0] = noise[0]
    
    # Apply AR(1) process
    for i in range(1, len(noise)):
        autocorrelated_noise[i] = 0.7 * autocorrelated_noise[i-1] + 0.3 * noise[i]
    
    # Add noise to data
    noisy_data = data * (1 + autocorrelated_noise)
    
    return noisy_data

# Utility function to validate Alpha Vantage API response
def validate_alpha_vantage_response(response):
    """
    Comprehensive validation for Alpha Vantage API responses.
    
    Parameters:
    -----------
    response : dict or requests.Response
        API response
        
    Returns:
    --------
    tuple
        (is_valid, message)
    """
    # Check HTTP response first if applicable
    if hasattr(response, 'status_code') and response.status_code != 200:
        return False, f"HTTP error: {response.status_code}"
    
    # Parse JSON if needed
    data = response.json() if hasattr(response, 'json') else response
    
    if not isinstance(data, dict):
        return False, "Invalid response format (not a dictionary)"
    
    # Check for explicit error messages
    if "Error Message" in data:
        return False, data["Error Message"]
    
    # Check for rate limit messages
    if "Note" in data and "call frequency" in data["Note"]:
        return False, f"Rate limit exceeded: {data['Note']}"
    
    # Check for different data structures based on endpoint
    
    # 1. Time series data
    time_series_key = next((k for k in data.keys() if "Time Series" in k), None)
    if time_series_key:
        if data.get(time_series_key) and len(data[time_series_key]) > 0:
            return True, f"Valid time series with {len(data[time_series_key])} data points"
        else:
            return False, "Time series is empty"
    
    # 2. Economic indicators
    if "data" in data and isinstance(data.get("data"), list):
        if len(data["data"]) > 0:
            return True, f"Valid economic data with {len(data['data'])} data points"
        else:
            return False, "Economic data list is empty"
    
    # 3. Forex data
    if "Realtime Currency Exchange Rate" in data:
        return True, "Valid real-time exchange rate data"
    
    # 4. Technical indicators
    indicator_key = next((k for k in data.keys() if "Technical Analysis:" in k), None)
    if indicator_key:
        if data.get(indicator_key) and len(data[indicator_key]) > 0:
            return True, f"Valid technical indicator with {len(data[indicator_key])} data points"
        else:
            return False, "Technical indicator data is empty"
    
    # 5. Global quote
    if "Global Quote" in data:
        if data["Global Quote"] and len(data["Global Quote"]) > 0:
            return True, "Valid global quote data"
        else:
            return False, "Global quote is empty"
    
    # 6. Sector performance
    if "Rank A: Real-Time Performance" in data:
        return True, "Valid sector performance data"
    
    # 7. Crypto data
    crypto_key = next((k for k in data.keys() if "Time Series (Digital Currency" in k), None)
    if crypto_key:
        if data.get(crypto_key) and len(data[crypto_key]) > 0:
            return True, f"Valid crypto data with {len(data[crypto_key])} data points"
        else:
            return False, "Crypto data is empty"
    
    # 8. News sentiment
    if "feed" in data:
        if data.get("feed") and len(data["feed"]) > 0:
            return True, f"Valid news feed with {len(data['feed'])} articles"
        else:
            return False, "News feed is empty"
    
    # Meta data check
    if "Meta Data" in data and len(data.keys()) > 1:
        # There's meta data and at least one more key, assume valid
        return True, "Response contains meta data and additional information"
    
    # If we can't determine specifically, check for non-empty structure
    if len(data.keys()) > 1:
        return True, "Response appears to contain valid data"
    else:
        return False, "Response contains no recognizable data structure"

# Decorator for caching Alpha Vantage API calls
def alpha_vantage_cache(cache_dir="./av_cache", ttl_hours=24):
    """
    Decorator for caching Alpha Vantage API calls with TTL.
    
    Parameters:
    -----------
    cache_dir : str
        Cache directory
    ttl_hours : int
        Time-to-live in hours
        
    Returns:
    --------
    function
        Decorated function
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create unique cache key based on function and params
            key_parts = [func.__name__]
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            cache_key = ''.join(c if c.isalnum() else '_' for c in "_".join(key_parts))
            cache_file = os.path.join(cache_dir, f"{cache_key}.json")
            
            # Try to load from cache
            if os.path.exists(cache_file):
                # Check if cache is still valid
                file_age = (time.time() - os.path.getmtime(cache_file)) / 3600
                if file_age < ttl_hours:
                    with open(cache_file, 'r') as f:
                        import json
                        return json.load(f)
            
            # Cache miss or expired - call function
            result = func(*args, **kwargs)
            
            # Cache the result (but only if it's not an error)
            if (isinstance(result, dict) and 
                "Error Message" not in result and
                "Note" not in result):
                with open(cache_file, 'w') as f:
                    import json
                    json.dump(result, f)
            
            return result
        return wrapper
    return decorator