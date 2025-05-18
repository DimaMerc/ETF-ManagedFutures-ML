# cuda_optimized_trend_predictor.py - Optimized with advanced CUDA enhancements

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import os
import pickle
import time
import logging
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
from simple_lstm_model import SimpleLSTM

# Import optimized components
from lstm_model import OptimizedLSTM, EnsembleTrendModel, TrendPredictionModule
from alpha_vantage_client_optimized import AlphaVantageClient, CommodityData
from error_handlers_and_utils import (
    NotEnoughDataError, validate_alpha_vantage_response,
    augment_time_series, time_warp, magnitude_scale, add_financial_noise
)

# Import configuration
from config import ASSETS, MODEL_PARAMS, CUDA_SETTINGS, PORTFOLIO_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CudaOptimizedTrendPredictor:
    """
    Enhanced trend predictor using CUDA optimizations for RTX 5090 GPUs
    with improved error handling and data management.
    """
    
    def __init__(self, api_key=None, returns_data=None, assets_dict=None, 
                 model_params=None, cuda_settings=None):
        """
        Initialize the optimized trend predictor.
        
        Parameters:
        -----------
        api_key : str, optional
            AlphaVantage API key
        returns_data : pd.DataFrame, optional
            Asset returns data (if not provided, will fetch using AlphaVantage)
        assets_dict : dict, optional
            Dictionary of asset classes and tickers
        model_params : dict, optional
            Parameters for the LSTM models
        cuda_settings : dict, optional
            CUDA optimization settings
        """
        self.api_key = api_key
        self.returns = returns_data
        self.assets_dict = assets_dict or ASSETS
        self.model_params = model_params or MODEL_PARAMS
        self.cuda_settings = cuda_settings or CUDA_SETTINGS
        
        # Initialize Alpha Vantage client with optimized implementation
        if api_key:
            self.av_client = AlphaVantageClient(api_key)
            self.commodity_client = CommodityData(self.av_client)
        else:
            self.av_client = None
            self.commodity_client = None
        
        # Initialize models, predictions, and data containers
        self._initialize_attributes()
        
        # Setup CUDA optimization for RTX 5090
        self._setup_cuda()
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('cache', exist_ok=True)
        
        # Load returns data if not provided
        if self.returns is None and self.av_client and self.assets_dict:
            self.load_returns_data()
    
    def _initialize_attributes(self):
        """Initialize model attributes and data containers."""
        # Initialize models and predictions
        self.trend_models = {}
        self.ensemble_models = {}
        self.scalers = {}
        self.cuda_graphs = {}
        self.trend_prediction_modules = {}
        
        # Initialize prediction attributes
        self.trend_predictions = None
        self.trend_direction = None
        self.trend_strength = None
        self.adjusted_direction = None
        self.adjusted_strength = None
        self.final_direction = None
        self.final_strength = None
        
        # Initialize data containers
        self.technical_indicators = {}
        self.economic_data = None
        self.news_sentiment = None
        
        # Initialize model performance metrics
        self.model_metrics = {}
        
        # Initialize data augmentation counters
        self.augmentation_stats = {
            'augmented_tickers': [],
            'total_augmented_data_points': 0
        }

    def _setup_cuda(self):
        """Set up CUDA optimizations based on settings."""
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  self.cuda_settings.get('enable_cuda', True) else 'cpu')
        self.use_cuda = self.device.type == 'cuda'
        
        if self.use_cuda:
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            gpu_capability = torch.cuda.get_device_capability(0)
            
            logger.info(f"CUDA enabled! Using {gpu_name} with {gpu_memory:.1f}GB memory")
            logger.info(f"GPU Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
            
            # Apply CUDA settings
            if self.cuda_settings.get('set_float32_precision') == 'high':
                torch.set_float32_matmul_precision('high')
                logger.info("Set float32 matmul precision to high")
                
            if self.cuda_settings.get('enable_cudnn_benchmark'):
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
                
            # Enable TF32 for Ampere (RTX 30xx) and later GPUs
            if gpu_capability[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for matrix operations")
            
            # Enable channels last memory format if specified
            if self.cuda_settings.get('enable_channels_last'):
                logger.info("Will use channels_last memory format for tensors")
                
            # Automatically select fastest algorithm
            torch.backends.cudnn.enabled = True
            
            # Pre-allocate memory for more efficient CUDA operation
            empty_tensor = torch.empty(100000000, device=self.device)  # 400MB preallocation
            del empty_tensor
            torch.cuda.empty_cache()
            
            # Reserve memory for persistent CUDA graph buffers if enabled
            if self.cuda_settings.get('enable_cuda_graphs', False):
                self.graph_memory_pool = []
                for _ in range(5):  # Reserve space for 5 graphs
                    tensor = torch.zeros(1000000, device=self.device)  # 4MB per reservation
                    self.graph_memory_pool.append(tensor)
        else:
            logger.info("CUDA not available. Using CPU for computations.")
        
        # Initialize mixed precision scaler if enabled
        self.use_mixed_precision = (self.cuda_settings.get('enable_mixed_precision', False) and 
                                   self.use_cuda)
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Enabled mixed precision training")
        else:
            self.scaler = None

    def load_returns_data(self, lookback_days=None):
        """
        Load returns data for all assets with special handling for FX pairs.
        
        Parameters:
        -----------
        lookback_days : int, optional
            Number of days to look back for returns data
            
        Returns:
        --------
        pd.DataFrame
            Returns data for all assets
        """
        if lookback_days is None:
            lookback_days = self.model_params.get('lookback_days', 365)
        
        logger.info("Loading returns data...")
        
        # Check if cached returns data exists and is not too old
        cache_file = "cached_returns.csv"
        cached_data_valid = False
        
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            hours_old = (time.time() - file_time) / 3600
            
            # Use cached data if less than 24 hours old
            if hours_old < 24:
                try:
                    # Load cached data
                    self.returns = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    logger.info(f"Using cached returns data ({hours_old:.1f} hours old)")
                    cached_data_valid = True
                    
                    # Check if we have all the assets we need
                    missing_assets = []
                    for asset_class, assets in self.assets_dict.items():
                        for asset in assets:
                            # For FX pairs, we need to handle the pair format
                            if '/' in asset:
                                asset_normalized = asset.replace('/', '_')
                                if asset_normalized not in self.returns.columns:
                                    missing_assets.append(asset)
                            elif asset not in self.returns.columns:
                                missing_assets.append(asset)
                    
                    # If we're missing assets, invalidate the cache
                    if missing_assets:
                        logger.info(f"Missing returns data for {len(missing_assets)} assets: {missing_assets}")
                        cached_data_valid = False
                except Exception as e:
                    logger.warning(f"Error loading cached returns: {str(e)}")
                    cached_data_valid = False
        
        # If no valid cached data, fetch from Alpha Vantage
        if not cached_data_valid:
            # Get returns data using the utility function or custom implementation
            all_returns = self._fetch_all_returns(lookback_days)
            
            # Cache the data
            if not all_returns.empty:
                all_returns.to_csv(cache_file)
                self.returns = all_returns
        
        # Make sure returns data is not empty before returning
        if self.returns is None or self.returns.empty:
            logger.error("Returns data is empty or None!")
            return None
        
        return self.returns
    
    

    def _fetch_all_returns(self, lookback_days):
        """
        Fetch returns data for all assets with special handling for different asset types.
        
        Parameters:
        -----------
        lookback_days : int
            Number of days to look back for returns data
                
        Returns:
        --------
        pd.DataFrame
            Returns data for all assets
        """
        all_returns = pd.DataFrame()
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Create commodity data handler if not already created
        if not hasattr(self, 'commodity_client'):
            self.commodity_client = CommodityData(self.av_client)
        
        # Process each asset class
        for asset_class, assets in self.assets_dict.items():
            logger.info(f"Processing {asset_class}...")
            
            for asset in assets:
                try:
                    # Handle different asset types
                    if asset_class == 'Commodities':
                        # Use commodity-specific endpoints for commodities
                        logger.info(f"Using commodity-specific endpoint for {asset}")
                        df = self.commodity_client.get_commodity_data(
                            commodity=asset,
                            interval='monthly',
                            cache=True
                        )
                        
                        # Find the column to use for returns
                        if df is not None and not df.empty:
                            returns_col = self._extract_returns_column(df)
                            if returns_col:
                                all_returns[asset] = df[returns_col]
                                logger.info(f"Added returns for {asset}")
                            else:
                                logger.warning(f"No suitable column found for returns in {asset}")
                        else:
                            logger.warning(f"No data found for commodity {asset}")
                    
                    elif asset_class == 'Currencies':
                        # For currencies, use the FX rate data directly
                        logger.info(f"Getting FX rate data for {asset}")
                        
                        # Get the currency pair components
                        from_currency, to_currency = self.get_currency_pair(asset)
                        
                        # Use get_fx_rate method to get FX data
                        df = self.av_client.get_fx_rate(
                            from_currency=from_currency,
                            to_currency=to_currency,
                            interval='monthly',  # Monthly data is more reliable
                            cache=True
                        )
                        
                        # Check if we have valid data
                        if df is not None and not df.empty:
                            if 'returns' in df.columns:
                                # Normalize the column name to avoid issues with slashes
                                asset_normalized = f"{from_currency}_{to_currency}"
                                all_returns[asset_normalized] = df['returns']
                                logger.info(f"Added returns for {from_currency}/{to_currency}")
                            else:
                                logger.warning(f"No returns column found for {from_currency}/{to_currency}")
                        else:
                            logger.warning(f"No FX data found for {from_currency}/{to_currency}")
                    
                    else:
                        # For standard assets, use the daily adjusted data
                        df = self.av_client.get_daily_adjusted(asset, cache=True)
                        
                        # Check if we have returns data
                        if not df.empty and 'returns' in df.columns:
                            # Filter by date if needed
                            if df.index[0] < pd.Timestamp(start_date):
                                df = df[df.index >= start_date]
                            
                            # Add to combined returns
                            all_returns[asset] = df['returns']
                            logger.info(f"Added returns for {asset}")
                        else:
                            logger.warning(f"No returns data found for {asset}")
                    
                except Exception as e:
                    logger.error(f"Error getting returns for {asset}: {str(e)}")
        
        # Align all returns to same dates
        all_returns = all_returns.fillna(method='ffill')
        
        return all_returns

    def _extract_returns_column(self, df):
        """Helper to extract the returns column from a dataframe."""
        if 'returns' in df.columns:
            return 'returns'
        elif 'value' in df.columns and df.shape[0] > 1:
            # Calculate returns if not present
            df['returns'] = df['value'].pct_change()
            return 'returns'
        elif 'close' in df.columns and df.shape[0] > 1:
            # Calculate returns from close
            df['returns'] = df['close'].pct_change()
            return 'returns'
        else:
            # Try to find any numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and df.shape[0] > 1:
                col_name = numeric_cols[0]
                df['returns'] = df[col_name].pct_change()
                return 'returns'
        return None

    def load_technical_indicators(self, use_cache=True, cache_file='cache/technical_indicators.pkl'):
        """
        Load technical indicators with improved caching and error handling.
        
        Parameters:
        -----------
        use_cache : bool, optional
            Whether to use cached data if available
        cache_file : str, optional
            Path to cache file
        
        Returns:
        --------
        dict
            Dictionary of technical indicators
        """
        logger.info("Loading technical indicators...")
        
        # Check cache first if enabled
        if use_cache and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 86400:  # Less than 1 day old
                logger.info(f"Using cached technical indicators ({cache_age/3600:.1f} hours old)")
                with open(cache_file, 'rb') as f:
                    self.technical_indicators = pickle.load(f)
                return self.technical_indicators
        
        # Initialize dictionary to store indicators
        technical_indicators = {}
        
        # Define indicators to fetch
        indicators = self.model_params.get('technical_indicators', [
            {'name': 'SMA', 'periods': [20, 50, 200]},
            {'name': 'EMA', 'periods': [20, 50, 200]},
            {'name': 'RSI', 'periods': [14]},
            {'name': 'MACD', 'periods': [12, 26, 9]},
            {'name': 'BBANDS', 'periods': [20]}
        ])
        
        start_time = time.time()
        
        # Process with ThreadPoolExecutor for parallel fetching
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(10, len(self.assets_dict) * len(indicators))
            
            # Define function for parallel execution
            def fetch_indicator(ticker, asset_class, indicator, period):
                try:
                    # Initialize ticker dict if not exists
                    if ticker not in technical_indicators:
                        technical_indicators[ticker] = {}
                    
                    # First attempt: try to get indicators from AlphaVantage
                    indicator_data = self.av_client.get_technical_indicator(
                        ticker, indicator['name'], time_period=period, cache=use_cache
                    )
                    
                    # Check if successful
                    if indicator_data is not None and not indicator_data.empty:
                        technical_indicators[ticker][f"{indicator['name']}_{period}"] = indicator_data
                        return True, ticker, indicator['name'], period
                    
                    # If failed, try local calculation
                    return self._calculate_indicator_locally(ticker, asset_class, 
                                                          indicator['name'], period, 
                                                          technical_indicators)
                        
                except Exception as e:
                    logger.warning(f"Error loading {indicator['name']}_{period} for {ticker}: {str(e)}")
                    # Try local calculation as fallback
                    return self._calculate_indicator_locally(ticker, asset_class, 
                                                          indicator['name'], period, 
                                                          technical_indicators)
            
            # Create tasks for parallel execution
            tasks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for asset_class, tickers in self.assets_dict.items():
                    for ticker in tickers:
                        for indicator in indicators:
                            for period in indicator['periods']:
                                # Submit task to executor
                                tasks.append(
                                    executor.submit(fetch_indicator, ticker, asset_class, 
                                                   indicator, period)
                                )
                
                # Process completed tasks
                success_count = 0
                fail_count = 0
                
                for future in as_completed(tasks):
                    try:
                        result = future.result()
                        if result[0]:  # Success
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        logger.error(f"Task error: {str(e)}")
                        fail_count += 1
                
                logger.info(f"Technical indicators: {success_count} successful, {fail_count} failed")
                
        except ImportError:
            # Fall back to sequential processing if ThreadPoolExecutor not available
            logger.info("ThreadPoolExecutor not available. Using sequential processing.")
            
            for asset_class, tickers in self.assets_dict.items():
                for ticker in tickers:
                    technical_indicators[ticker] = {}
                    
                    for indicator in indicators:
                        for period in indicator['periods']:
                            try:
                                # First attempt: try to get indicators from AlphaVantage
                                indicator_data = self.av_client.get_technical_indicator(
                                    ticker, indicator['name'], time_period=period, cache=use_cache
                                )
                                
                                # Check if successful
                                if indicator_data is not None and not indicator_data.empty:
                                    technical_indicators[ticker][f"{indicator['name']}_{period}"] = indicator_data
                                    
                                else:
                                    # If failed, try local calculation
                                    self._calculate_indicator_locally(ticker, asset_class, 
                                                                   indicator['name'], period, 
                                                                   technical_indicators)
                                    
                            except Exception as e:
                                logger.warning(f"Error loading {indicator['name']}_{period} for {ticker}: {str(e)}")
                                # Try local calculation as fallback
                                self._calculate_indicator_locally(ticker, asset_class, 
                                                               indicator['name'], period, 
                                                               technical_indicators)
        
        # Cache the results
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(technical_indicators, f)
        
        self.technical_indicators = technical_indicators
        
        elapsed = time.time() - start_time
        indicator_count = sum(len(v) for v in technical_indicators.values())
        
        logger.info(f"Loaded {indicator_count} technical indicators for {len(technical_indicators)} assets in {elapsed:.1f} seconds")
        
        return technical_indicators
    
    def _calculate_indicator_locally(self, ticker, asset_class, indicator_name, period, indicators_dict):
        """
        Calculate technical indicators locally when API fails.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        asset_class : str
            Asset class (Commodities, Currencies, etc.)
        indicator_name : str
            Indicator name (SMA, EMA, RSI, etc.)
        period : int
            Indicator period
        indicators_dict : dict
            Dictionary to store results
            
        Returns:
        --------
        tuple
            (success, ticker, indicator_name, period)
        """
        try:
            # Get price data based on asset class
            if asset_class == 'Commodities':
                df = self.commodity_client.get_commodity_data(ticker)
                price_series = df['close'] if 'close' in df.columns else df['value']
            elif asset_class == 'Currencies':
                df = self.av_client.get_fx_rate(ticker)
                price_series = df['close']
            elif ticker == 'TREASURY_YIELD':
                treasury_data = self.av_client.get_economic_indicator('TREASURY_YIELD')
                price_series = treasury_data['10year'] if '10year' in treasury_data.columns else None
                if price_series is None:
                    return False, ticker, indicator_name, period
            else:
                df = self.av_client.get_daily_adjusted(ticker)
                price_series = df['close']
            
            # Calculate indicators based on type
            if indicator_name == 'SMA':
                result = price_series.rolling(window=period).mean()
                indicators_dict[ticker][f"SMA_{period}"] = pd.DataFrame({f"SMA_{period}": result})
                
            elif indicator_name == 'EMA':
                result = price_series.ewm(span=period, adjust=False).mean()
                indicators_dict[ticker][f"EMA_{period}"] = pd.DataFrame({f"EMA_{period}": result})
                
            elif indicator_name == 'RSI':
                # Calculate RSI
                delta = price_series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                indicators_dict[ticker][f"RSI_{period}"] = pd.DataFrame({f"RSI_{period}": rsi})
                
            elif indicator_name == 'MACD':
                # Calculate MACD with standard parameters
                ema12 = price_series.ewm(span=12, adjust=False).mean()
                ema26 = price_series.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                
                indicators_dict[ticker]["MACD_12_26_9"] = pd.DataFrame({
                    'MACD': macd_line,
                    'MACD_Signal': signal_line,
                    'MACD_Hist': macd_line - signal_line
                })
            
            elif indicator_name == 'BBANDS':
                # Calculate Bollinger Bands
                sma = price_series.rolling(window=period).mean()
                std = price_series.rolling(window=period).std()
                
                upper_band = sma + 2 * std
                lower_band = sma - 2 * std
                
                indicators_dict[ticker][f"BBANDS_{period}"] = pd.DataFrame({
                    'Real Middle Band': sma,
                    'Real Upper Band': upper_band,
                    'Real Lower Band': lower_band
                })
            
            return True, ticker, indicator_name, period
            
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}_{period} for {ticker}: {str(e)}")
            return False, ticker, indicator_name, period
        
    
    def is_currency_symbol(self, symbol):
        """Check if a symbol is a currency."""
        currency_codes = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']
        return symbol in currency_codes or '/' in symbol

    def get_currency_pair(self, symbol):
        """Convert currency symbol to proper FX pair format."""
        if '/' in symbol:
            # Already in pair format
            return symbol.split('/')
        elif symbol in ['JPY', 'CAD', 'CHF']:
            # These are typically quoted as USD/XXX
            return 'USD', symbol
        else:
            # Other major currencies are typically quoted as XXX/USD
            return symbol, 'USD'
        
    def load_economic_data(self, use_cache=True, cache_file='cache/economic_data.pkl'):
        """
        Load economic indicator data with enhanced caching and validation.
        
        Parameters:
        -----------
        use_cache : bool, optional
            Whether to use cached data if available
        cache_file : str, optional
            Path to cache file
        
        Returns:
        --------
        pd.DataFrame
            Economic indicator data
        """
        logger.info("Loading economic data...")
        
        # Check cache first if enabled
        if use_cache and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 86400 * 7:  # Less than 7 days old (economic data updates less frequently)
                logger.info(f"Using cached economic data ({cache_age/86400:.1f} days old)")
                with open(cache_file, 'rb') as f:
                    self.economic_data = pickle.load(f)
                return self.economic_data
        
        # Define economic indicators to fetch
        indicators = self.model_params.get('economic_indicators', [
            'REAL_GDP', 'TREASURY_YIELD', 'CPI', 'INFLATION', 
            'RETAIL_SALES', 'UNEMPLOYMENT', 'INDUSTRIAL_PRODUCTION',
            'HOUSING_STARTS', 'CONSUMER_SENTIMENT'
        ])
        
        start_time = time.time()
        economic_data = pd.DataFrame()
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Define function for parallel execution
            def fetch_economic_indicator(indicator):
                try:
                    # Fetch indicator data
                    indicator_data = self.av_client.get_economic_indicator(indicator)
                    
                    if indicator_data is not None and not indicator_data.empty:
                        if indicator == 'TREASURY_YIELD':
                            # For treasury yield, keep all maturity columns
                            renamed_cols = {col: f"{indicator}_{col}" for col in indicator_data.columns}
                            return indicator_data.rename(columns=renamed_cols)
                        else:
                            # For other indicators, use indicator name as prefix
                            renamed_cols = {col: f"{indicator}_{col}" for col in indicator_data.columns}
                            return indicator_data.rename(columns=renamed_cols)
                    
                    return None
                    
                except Exception as e:
                    logger.error(f"Error loading economic indicator {indicator}: {str(e)}")
                    return None
            
            # Fetch indicators in parallel
            indicators_data = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit tasks
                future_to_indicator = {executor.submit(fetch_economic_indicator, indicator): indicator 
                                     for indicator in indicators}
                
                # Process completed tasks
                for future in as_completed(future_to_indicator):
                    indicator = future_to_indicator[future]
                    try:
                        indicator_data = future.result()
                        if indicator_data is not None:
                            indicators_data.append(indicator_data)
                            logger.info(f"Loaded economic indicator: {indicator}")
                    except Exception as e:
                        logger.error(f"Error processing {indicator}: {str(e)}")
            
            # Combine all indicators into a single dataframe
            if indicators_data:
                # Find common date range for all dataframes
                all_dates = set()
                for df in indicators_data:
                    all_dates.update(df.index)
                
                all_dates = sorted(all_dates)
                
                # Create a dataframe with all dates
                economic_data = pd.DataFrame(index=all_dates)
                
                # Merge each indicator dataframe
                for df in indicators_data:
                    # Reindex to all dates
                    reindexed = df.reindex(all_dates)
                    
                    # Add each column
                    for col in df.columns:
                        economic_data[col] = reindexed[col]
                
                # Ensure proper datetime index
                economic_data.index = pd.to_datetime(economic_data.index)
                economic_data = economic_data.sort_index()
                
                # Forward fill missing values
                economic_data = economic_data.ffill().bfill()
        
        except ImportError:
            # Fall back to sequential processing
            logger.info("ThreadPoolExecutor not available. Using sequential processing.")
            
            for indicator in indicators:
                try:
                    # Fetch indicator data
                    indicator_data = self.av_client.get_economic_indicator(indicator)
                    
                    if indicator_data is not None and not indicator_data.empty:
                        if indicator == 'TREASURY_YIELD':
                            # For treasury yield, keep all maturity columns
                            renamed_cols = {col: f"{indicator}_{col}" for col in indicator_data.columns}
                            indicator_data = indicator_data.rename(columns=renamed_cols)
                        else:
                            # For other indicators, use indicator name as prefix
                            renamed_cols = {col: f"{indicator}_{col}" for col in indicator_data.columns}
                            indicator_data = indicator_data.rename(columns=renamed_cols)
                        
                        # Add to economic data - handle empty dataframe case
                        if economic_data.empty:
                            economic_data = indicator_data
                        else:
                            # Merge with different indices
                            economic_data = economic_data.join(indicator_data, how='outer')
                        
                        logger.info(f"Loaded economic indicator: {indicator}")
                    
                except Exception as e:
                    logger.error(f"Error loading economic indicator {indicator}: {str(e)}")
        
        # Handle missing values and reindex
        if not economic_data.empty:
            # Forward fill missing values
            economic_data = economic_data.ffill()
            
            # Cache the results
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(economic_data, f)
        
        self.economic_data = economic_data
        
        elapsed = time.time() - start_time
        logger.info(f"Loaded economic data with {len(economic_data.columns)} indicators in {elapsed:.1f} seconds")
        
        return economic_data
    
    def load_news_sentiment(self, use_cache=True, cache_file='cache/news_sentiment.pkl'):
        """
        Load news sentiment data with improved batching and error handling.
        
        Parameters:
        -----------
        use_cache : bool, optional
            Whether to use cached data if available
        cache_file : str, optional
            Path to cache file
        
        Returns:
        --------
        pd.DataFrame
            News sentiment data
        """
        logger.info("Loading news sentiment data...")
        
        # Check cache first if enabled
        if use_cache and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 86400:  # Less than 1 day old (news updates frequently)
                logger.info(f"Using cached news sentiment data ({cache_age/3600:.1f} hours old)")
                with open(cache_file, 'rb') as f:
                    self.news_sentiment = pickle.load(f)
                return self.news_sentiment
        
        start_time = time.time()
        sentiment_data_frames = []
        
        # Get all tickers from assets_dict
        all_tickers = []
        for asset_class, tickers in self.assets_dict.items():
            all_tickers.extend(tickers)
        
        # Process in smaller batches of 5 tickers to avoid API limitations
        batch_size = 5
        batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Define function for parallel execution
            def fetch_sentiment_batch(batch):
                try:
                    batch_str = ','.join(batch)
                    batch_sentiment = self.av_client.get_news_sentiment(tickers=batch_str)
                    if not batch_sentiment.empty:
                        logger.info(f"Loaded sentiment data for batch: {batch_str}")
                        return batch_sentiment
                    return None
                except Exception as e:
                    logger.error(f"Error loading sentiment for batch: {str(e)}")
                    return None
            
            # Process batches in parallel with delay between batches
            with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 workers to avoid rate limits
                future_to_batch = {}
                
                # Submit first batch immediately
                if batches:
                    future_to_batch[executor.submit(fetch_sentiment_batch, batches[0])] = 0
                
                # Process completed tasks and submit new ones with delay
                batch_idx = 1
                while future_to_batch or batch_idx < len(batches):
                    # Process completed futures
                    completed_futures = []
                    for future in list(future_to_batch.keys()):
                        if future.done():
                            idx = future_to_batch[future]
                            try:
                                result = future.result()
                                if result is not None:
                                    sentiment_data_frames.append(result)
                                    logger.info(f"Processed batch {idx+1}/{len(batches)}")
                            except Exception as e:
                                logger.error(f"Error processing batch {idx+1}: {str(e)}")
                            completed_futures.append(future)
                    
                    # Remove completed futures
                    for future in completed_futures:
                        del future_to_batch[future]
                    
                    # Submit new batch if available
                    if batch_idx < len(batches):
                        future_to_batch[executor.submit(fetch_sentiment_batch, batches[batch_idx])] = batch_idx
                        batch_idx += 1
                        # Add delay to avoid rate limits
                        time.sleep(2.0)
                    
                    # Slight pause to avoid busy waiting
                    if future_to_batch:
                        time.sleep(0.1)
            
            # Try loading economic topics separately
            economic_topics = ['economy', 'inflation', 'monetary_policy', 'fiscal_policy', 
                             'trade', 'interest_rates', 'recession', 'economic_growth']
            
            with ThreadPoolExecutor(max_workers=1) as executor:  # Limit to 1 worker for topics
                future_to_topic = {}
                
                # Submit first topic
                if economic_topics:
                    future_to_topic[executor.submit(
                        lambda t: self.av_client.get_news_sentiment(topics=t),
                        economic_topics[0]
                    )] = 0
                
                # Process completed tasks and submit new ones with delay
                topic_idx = 1
                while future_to_topic or topic_idx < len(economic_topics):
                    # Process completed futures
                    completed_futures = []
                    for future in list(future_to_topic.keys()):
                        if future.done():
                            idx = future_to_topic[future]
                            try:
                                result = future.result()
                                if result is not None and not result.empty:
                                    sentiment_data_frames.append(result)
                                    logger.info(f"Processed topic {economic_topics[idx]}")
                            except Exception as e:
                                logger.error(f"Error processing topic {economic_topics[idx]}: {str(e)}")
                            completed_futures.append(future)
                    
                    # Remove completed futures
                    for future in completed_futures:
                        del future_to_topic[future]
                    
                    # Submit new topic if available
                    if topic_idx < len(economic_topics):
                        future_to_topic[executor.submit(
                            lambda t: self.av_client.get_news_sentiment(topics=t),
                            economic_topics[topic_idx]
                        )] = topic_idx
                        topic_idx += 1
                        # Add delay to avoid rate limits
                        time.sleep(3.0)
                    
                    # Slight pause to avoid busy waiting
                    if future_to_topic:
                        time.sleep(0.1)
        
        except ImportError:
            # Fall back to sequential processing
            logger.info("ThreadPoolExecutor not available. Using sequential processing.")
            
            for i, batch in enumerate(batches):
                try:
                    batch_str = ','.join(batch)
                    batch_sentiment = self.av_client.get_news_sentiment(tickers=batch_str)
                    if not batch_sentiment.empty:
                        sentiment_data_frames.append(batch_sentiment)
                        logger.info(f"Loaded sentiment data for batch {i+1}/{len(batches)}")
                    
                    # Add brief delay to avoid API rate limits
                    time.sleep(2.0)
                except Exception as e:
                    logger.error(f"Error loading sentiment for batch {i+1}: {str(e)}")
            
            # Try loading economic topics separately
            economic_topics = ['economy', 'inflation', 'monetary_policy', 'fiscal_policy', 
                             'trade', 'interest_rates', 'recession', 'economic_growth']
            
            for topic in economic_topics:
                try:
                    topic_sentiment = self.av_client.get_news_sentiment(topics=topic)
                    if not topic_sentiment.empty:
                        sentiment_data_frames.append(topic_sentiment)
                        logger.info(f"Loaded sentiment data for topic: {topic}")
                    
                    # Add brief delay to avoid API rate limits
                    time.sleep(3.0)
                except Exception as e:
                    logger.error(f"Error loading sentiment for topic {topic}: {str(e)}")
        
        # Combine all sentiment data
        combined_sentiment = pd.DataFrame()
        if sentiment_data_frames:
            # Concatenate all dataframes
            combined_sentiment = pd.concat(sentiment_data_frames, ignore_index=True)
            
            # Convert date to datetime if needed
            if 'date' in combined_sentiment.columns:
                combined_sentiment['date'] = pd.to_datetime(combined_sentiment['date'])
                
            # Cache the results
            if not combined_sentiment.empty:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(combined_sentiment, f)
        
        self.news_sentiment = combined_sentiment
        
        elapsed = time.time() - start_time
        logger.info(f"Loaded sentiment data with {len(combined_sentiment)} records in {elapsed:.1f} seconds")
        
        return combined_sentiment
    
    def prepare_lstm_data(self, series, seq_length, augment_if_needed=True):
        """
        Prepare data for LSTM model with CUDA optimization and data augmentation.
        
        Parameters:
        -----------
        series : np.array
            Time series data
        seq_length : int
            Sequence length
        augment_if_needed : bool
            Whether to augment data if not enough samples
            
        Returns:
        --------
        tuple
            (X, y) tensors for model training
        """
        original_len = len(series)
        
        # Ensure data is not too short
        min_samples_needed = seq_length + 30  # Need at least this many points for training
        
        # Apply data augmentation if needed and requested
        if augment_if_needed and original_len < min_samples_needed:
            logger.info(f"Not enough data points ({original_len}) for sequence length {seq_length}. Augmenting data.")
            
            # Use augmentation techniques from error_handlers_and_utils
            augmented_series = augment_time_series(series, augmentation_factor=3)
            
            # Track augmentation for reporting
            if hasattr(self, 'augmentation_stats'):
                self.augmentation_stats['augmented_tickers'].append(f"Unknown ({original_len} points)")
                self.augmentation_stats['total_augmented_data_points'] += len(augmented_series) - original_len
            
            # Use augmented data
            series = augmented_series
            logger.info(f"Data augmented to {len(series)} points")
        
        # Create sequences in memory-efficient way
        X, y = [], []
        for i in range(len(series) - seq_length):
            X.append(series[i:i+seq_length])
            y.append(series[i+seq_length])
        
        # Convert to numpy arrays
        X_np = np.array(X)
        y_np = np.array(y)
        
        # Convert to torch tensors with optimized memory layout
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
            
        # Pin memory if requested (must be done after tensor creation from numpy)
        if self.use_cuda and self.cuda_settings.get('pin_memory', False):
            X_tensor = X_tensor.pin_memory()
            y_tensor = y_tensor.pin_memory()
            
        # Move to device
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        # Standard memory layout - channels_last only works for 4D tensors
        X_tensor = X_tensor.view(X_tensor.shape[0], X_tensor.shape[1], 1)

        return X_tensor, y_tensor

    def build_trend_models(self, force_rebuild=False):
        """
        Build LSTM-based trend prediction models with CUDA optimization.
        Now with ensemble model creation enabled.
        
        Parameters:
        -----------
        force_rebuild : bool, optional
            Whether to force rebuilding models even if they already exist
        
        Returns:
        --------
        dict
            Dictionary of trend models
        """
        logger.info("Building trend prediction models with CUDA optimizations...")
        
        # Ensure we have returns data
        if self.returns is None:
            if self.av_client and self.assets_dict:
                self.load_returns_data()
            else:
                logger.error("No returns data available. Load data first.")
                return None
        
        # Create model directory
        os.makedirs('models', exist_ok=True)
        
        # Track statistics
        n_built = 0
        n_skipped = 0
        n_failed = 0
        n_augmented = 0
        build_times = []
        
        # Reset augmentation stats
        self.augmentation_stats = {
            'augmented_tickers': [],
            'total_augmented_data_points': 0
        }
        
        # Get sequence length from model parameters
        seq_length = self.model_params.get('sequence_length', 60)
        
        # Get returns dataframe
        returns = self.returns

        try:
            # Log the shape of the returns DataFrame for debugging
            logger.info(f"Returns DataFrame shape: {returns.shape}")
            logger.info(f"Returns DataFrame columns: {returns.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error inspecting returns: {str(e)}")
        
        # Build a model for each asset
        for ticker in returns.columns:
            # Skip if model already exists and not force rebuilding
            model_exists = ticker in self.trend_models and not force_rebuild
            if model_exists:
                logger.info(f"  Model for {ticker} already exists. Skipping.")
                n_skipped += 1
                continue
            
            try:
                logger.info(f"  Building model for {ticker}...")
                start_time = time.time()
                
                try:
                    series = returns[ticker].dropna().values
                    logger.info(f"  Data for {ticker}: {len(series)} points, "
                                f"range [{series.min():.4f}, {series.max():.4f}]")
                except Exception as e:
                    logger.error(f"  Error preparing data for {ticker}: {str(e)}")
                    n_failed += 1
                    continue
                
                # Check if we have enough data
                min_required = seq_length + 20  # At least 20 points for testing
                
                if len(series) < min_required:
                    logger.warning(f"  Not enough data for {ticker} (have {len(series)}, need > {min_required})")
                    
                    # Try data augmentation
                    logger.info(f"  Attempting data augmentation for {ticker}")
                    try:
                        augmented_series = augment_time_series(series, augmentation_factor=4)
                        
                        if len(augmented_series) >= min_required:
                            logger.info(f"  Successfully augmented data for {ticker} from {len(series)} to {len(augmented_series)} points")
                            series = augmented_series
                            n_augmented += 1
                            
                            # Track augmentation statistics
                            self.augmentation_stats['augmented_tickers'].append(ticker)
                            self.augmentation_stats['total_augmented_data_points'] += len(augmented_series) - len(series)
                        else:
                            logger.error(f"  Augmentation failed to produce enough data points for {ticker}")
                            n_failed += 1
                            continue
                    except Exception as e:
                        logger.error(f"  Augmentation error for {ticker}: {str(e)}")
                        n_failed += 1
                        continue
                
                # Scale data to [-1, 1] range
                try:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
                    
                    # Store scaler for later use
                    self.scalers[ticker] = scaler
                except Exception as e:
                    logger.error(f"  Error scaling data for {ticker}: {str(e)}")
                    n_failed += 1
                    continue
                
                # Prepare data for LSTM with optimized memory layout
                X, y = self.prepare_lstm_data(series_scaled, seq_length, augment_if_needed=False)
                
                # Split into train and validation
                try:
                    split = int(0.85 * len(X))  # Increased train proportion
                    X_train, X_val = X[:split], X[split:]
                    y_train, y_val = y[:split], y[split:]
                    logger.info(f"  Split data: {len(X_train)} train, {len(X_val)} validation examples")
                except Exception as e:
                    logger.error(f"  Error splitting data for {ticker}: {str(e)}")
                    n_failed += 1
                    continue
                
                # Get model hyperparameters
                hidden_sizes = self.model_params.get('hidden_sizes', [128, 64, 32])
                num_layers = self.model_params.get('num_layers', 3)
                dropout = self.model_params.get('dropout', 0.3)
                
                # Initialize optimized LSTM model
                try:
                    logger.info(f"  Creating SimpleLSTM model for {ticker}")
                    model = SimpleLSTM(
                        input_size=1,
                        hidden_size=64,
                        num_layers=2,
                        output_size=1
                    ).to(self.device)
                    logger.info(f"  Model created successfully and moved to {self.device}")
                except Exception as e:
                    logger.error(f"  Error creating model for {ticker}: {str(e)}")
                    n_failed += 1
                    continue
                
                # Test forward pass with sample data for debugging
                try:
                    logger.info(f"  Testing forward pass with sample data")
                    sample_input = X_train[:1]  # Take first sample
                    with torch.no_grad():
                        output = model(sample_input)
                        # Check what the model returns
                        if isinstance(output, tuple):
                            logger.info(f"  Forward pass returns a tuple with {len(output)} elements")
                            for i, item in enumerate(output):
                                logger.info(f"    Element {i}: shape {item.shape}, type {type(item)}")
                        else:
                            logger.info(f"  Forward pass returns a single tensor of shape {output.shape}")
                except Exception as e:
                    logger.error(f"  Forward pass test failed for {ticker}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    n_failed += 1
                    continue

                # Create training module with mixed precision support
                trend_module = TrendPredictionModule(
                    model=model,
                    learning_rate=self.model_params.get('learning_rate', 0.001),
                    weight_decay=self.model_params.get('weight_decay', 1e-5),
                    device=self.device,
                    use_custom_loss=True  # Enable the custom loss function
                )
                
                # Create simple data loaders
                class SimpleDataLoader:
                    def __init__(self, X, y, batch_size):
                        self.X = X
                        self.y = y
                        self.batch_size = batch_size
                        self.n_samples = X.shape[0]
                        
                    def __iter__(self):
                        indices = torch.randperm(self.n_samples)
                        for i in range(0, self.n_samples, self.batch_size):
                            batch_indices = indices[i:i+self.batch_size]
                            yield self.X[batch_indices], self.y[batch_indices]
                            
                    def __len__(self):
                        return (self.n_samples + self.batch_size - 1) // self.batch_size
                
                # Calculate optimal batch size
                if self.use_cuda:
                    batch_size = trend_module.optimize_batch_size(
                        seq_length=seq_length,
                        feature_dim=1,
                        target_memory=0.7  # Target memory utilization
                    )
                else:
                    batch_size = min(32, len(X_train))
                
                # Create data loaders
                train_loader = SimpleDataLoader(X_train, y_train, batch_size)
                val_loader = SimpleDataLoader(X_val, y_val, batch_size)
                
                # Train the model with early stopping
                train_losses, val_losses, best_val_loss = trend_module.train_epochs(
                    train_loader,
                    val_loader,
                    epochs=self.model_params.get('epochs', 50),
                    accumulation_steps=4,  # Gradient accumulation for stability
                    patience=self.model_params.get('early_stopping_patience', 10)
                )
                
                # Create CUDA Graph for inference if supported
                sample_input = torch.zeros(
                    (1, seq_length, 1), 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                if self.use_cuda and self.cuda_settings.get('enable_cuda_graphs', False):
                    try:
                        # Create CUDA graph
                        cuda_graph, static_input, static_outputs = trend_module.create_cuda_graph(sample_input)
                        self.cuda_graphs[ticker] = (cuda_graph, static_input, static_outputs)
                        logger.info(f"  Created CUDA graph for {ticker} model inference")
                    except Exception as e:
                        logger.warning(f"  CUDA graph creation failed for {ticker}: {str(e)}")
                        self.cuda_graphs[ticker] = (None, None, None)
                
                # Store model and training module
                self.trend_models[ticker] = model
                self.trend_prediction_modules[ticker] = trend_module
                
                # Store training metrics
                self.model_metrics[ticker] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'data_points': len(series),
                    'augmented': ticker in self.augmentation_stats['augmented_tickers']
                }
                
                # Track build time
                build_time = time.time() - start_time
                build_times.append(build_time)
                logger.info(f"  Built model for {ticker} in {build_time:.1f} seconds")
                
                n_built += 1
                
                # Save model after each successful build
                self.save_model(ticker)
                
            except Exception as e:
                logger.error(f"  Error building model for {ticker}: {str(e)}")
                n_failed += 1
                continue
        
        # Build ensemble models if enough individual models
        if len(self.trend_models) >= 3:
            # Now enable ensemble models
            logger.info("Building ensemble models...")
            self._build_ensemble_models()
        
        # Print summary
        if build_times:
            avg_build_time = sum(build_times) / len(build_times)
            logger.info(f"Built {n_built} models, skipped {n_skipped}, failed {n_failed}, augmented {n_augmented}")
            logger.info(f"Average build time: {avg_build_time:.1f} seconds per model")
            
            if n_augmented > 0:
                logger.info(f"Data augmentation applied to {n_augmented} assets")
                logger.info(f"Added {self.augmentation_stats['total_augmented_data_points']} synthetic data points")
        
        return self.trend_models
    
    def _build_ensemble_models(self):
        """
        Build ensemble models for each asset class.
        Simplified version compatible with SimpleLSTM models.
        """
        logger.info("Building ensemble models for asset classes...")
        
        # Dictionary to store ensemble models by asset class
        self.ensemble_models = {}
        
        # Process each asset class
        for asset_class, tickers in self.assets_dict.items():
            # Find tickers that have models
            available_tickers = [t for t in tickers if t in self.trend_models]
            
            # Need at least 3 models to build an ensemble
            if len(available_tickers) >= 3:
                logger.info(f"Building ensemble for {asset_class} with {len(available_tickers)} models")
                
                # Select models to use in ensemble (up to 5)
                ensemble_size = min(5, len(available_tickers))
                selected_tickers = available_tickers[:ensemble_size]
                
                # Store ensemble configuration
                self.ensemble_models[asset_class] = {
                    'tickers': selected_tickers,
                    'weights': [1.0/ensemble_size] * ensemble_size  # Equal weights initially
                }
                
                # Log the models included in the ensemble
                logger.info(f"Ensemble for {asset_class} includes: {', '.join(selected_tickers)}")
            else:
                logger.info(f"Not enough models for {asset_class} ensemble (need at least 3, have {len(available_tickers)})")
        
        logger.info(f"Built {len(self.ensemble_models)} ensemble models")
        
        return self.ensemble_models
    
    def predict_with_ensemble(self, asset_class, series=None):
        """
        Generate predictions using an ensemble of models for a specific asset class.
        
        Parameters:
        -----------
        asset_class : str
            Asset class to predict
        series : pd.DataFrame, optional
            Returns data to use (if None, uses self.returns)
            
        Returns:
        --------
        tuple
            (prediction, direction, strength)
        """
        # Check if ensemble exists for this asset class
        if asset_class not in self.ensemble_models:
            logger.warning(f"No ensemble model found for {asset_class}")
            return None, 0, 0
        
        # Get ensemble configuration
        ensemble_config = self.ensemble_models[asset_class]
        ensemble_tickers = ensemble_config['tickers']
        ensemble_weights = ensemble_config['weights']
        
        # Generate predictions from individual models
        predictions = []
        directions = []
        strengths = []
        
        for ticker in ensemble_tickers:
            # Use existing model to predict
            pred, direction, strength, _ = self.predict_trend(ticker, series)
            
            if pred is not None:
                predictions.append(pred)
                directions.append(direction)
                strengths.append(strength)
            else:
                # If any model fails, use neutral prediction
                predictions.append(0)
                directions.append(0)
                strengths.append(0)
        
        # Check if we have valid predictions
        if not predictions:
            logger.warning(f"No valid predictions for {asset_class} ensemble")
            return None, 0, 0
        
        # Apply weighted average
        weighted_pred = sum(p * w for p, w in zip(predictions, ensemble_weights))
        
        # For direction, use majority vote weighted by strength and model weight
        direction_strength = sum(d * s * w for d, s, w in zip(directions, strengths, ensemble_weights))
        ensemble_direction = 1 if direction_strength > 0 else (-1 if direction_strength < 0 else 0)
        
        # Strength is the absolute value of the weighted prediction
        ensemble_strength = abs(weighted_pred)
        
        logger.info(f"Ensemble prediction for {asset_class}: {weighted_pred:.4f} (direction: {ensemble_direction}, strength: {ensemble_strength:.4f})")
        
        return weighted_pred, ensemble_direction, ensemble_strength
        
    def save_model(self, ticker):
        """
        Save a single model to disk with all related components.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        bool
            Success status
        """
        if ticker not in self.trend_models:
            logger.warning(f"No model found for {ticker}")
            return False
        
        # Create ticker directory
        ticker_dir = os.path.join('models', ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        try:
            # Save model weights
            model = self.trend_models[ticker]
            model_file = os.path.join(ticker_dir, "lstm_model.pt")
            torch.save(model.state_dict(), model_file)
            
            # Save metadata including model architecture
            metadata_file = os.path.join(ticker_dir, "metadata.pkl")
            metadata = {
                'model_type': 'SimpleLSTM',
                'hidden_size': model.hidden_size,  # Now a single value
                'num_layers': model.num_layers,
                'sequence_length': self.model_params.get('sequence_length', 60),
                'build_time': datetime.now().isoformat(),
                'metrics': self.model_metrics.get(ticker, {})
            }
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save scaler if available
            if ticker in self.scalers:
                scaler_file = os.path.join(ticker_dir, "scaler.pkl")
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers[ticker], f)
            
            logger.info(f"Saved model and metadata for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {ticker}: {str(e)}")
            return False
    
    def save_trend_models(self):
        """Save all trend prediction models with updated ensemble saving."""
        logger.info("Saving all trend prediction models...")
        
        # Track success and failure counts
        success_count = 0
        failure_count = 0
        
        # Save individual models
        for ticker in self.trend_models.keys():
            if self.save_model(ticker):
                success_count += 1
            else:
                failure_count += 1
        
        # Save ensemble models with modified approach
        for asset_class, ensemble_info in self.ensemble_models.items():
            try:
                class_dir = os.path.join('models', f"ensemble_{asset_class}")
                os.makedirs(class_dir, exist_ok=True)
                
                # Save metadata - now includes weights directly since we don't have a model object
                metadata_file = os.path.join(class_dir, "metadata.pkl")
                metadata = {
                    'tickers': ensemble_info['tickers'],
                    'weights': ensemble_info.get('weights', [1/len(ensemble_info['tickers'])] * len(ensemble_info['tickers'])),
                    'build_time': datetime.now().isoformat()
                }
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
                
                success_count += 1
                logger.info(f"Saved ensemble configuration for {asset_class}")
                
            except Exception as e:
                logger.error(f"Error saving ensemble model for {asset_class}: {str(e)}")
                failure_count += 1
        
        logger.info(f"Saved {success_count} models successfully, {failure_count} failed")
        return success_count, failure_count
    
    def load_model(self, ticker):
        """
        Load a pre-trained model for a specific ticker with improved error handling.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        bool
            Whether the model was loaded successfully
        """
        model_dir = os.path.join('models', ticker)
        
        if not os.path.exists(model_dir):
            logger.warning(f"No saved model found for {ticker}")
            return False
        
        try:
            # Create model with fixed architecture that matches saved models
            model = SimpleLSTM(
                input_size=1,
                hidden_size=64,      # Fixed to match saved model
                num_layers=2,        # Fixed to match saved model
                output_size=1,
                dropout=0.2          # Fixed to match saved model
            )
            
            # Load model state
            model_path = os.path.join(model_dir, "model.pt")
            if os.path.exists(model_path):
                # Load to CPU first to avoid CUDA issues
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Handle potential key mismatches in state dict
                # This helps with compatibility if model definition has changed slightly
                model_dict = model.state_dict()
                # Filter out incompatible keys
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                # Update model dict with filtered state dict
                model_dict.update(state_dict)
                # Load the filtered state dict
                model.load_state_dict(model_dict, strict=False)
                
                # Move to appropriate device
                model = model.to(self.device)
                
                # Create prediction module wrapper
                trend_module = TrendPredictionModule(
                    model=model,
                    learning_rate=0.001,
                    weight_decay=1e-5,
                    device=self.device
                )
                
                # Store model and prediction module
                self.trend_models[ticker] = model
                self.trend_prediction_modules[ticker] = trend_module
                
                # Load model metadata
                metadata_path = os.path.join(model_dir, "metadata.pkl")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.model_metrics[ticker] = pickle.load(f)
                
                # Try to load CUDA graph information if available
                cuda_graph_path = os.path.join(model_dir, "cuda_graph.pt")
                if os.path.exists(cuda_graph_path) and self.use_cuda:
                    try:
                        cuda_graph_info = torch.load(cuda_graph_path, map_location=self.device)
                        if cuda_graph_info and len(cuda_graph_info) == 3:
                            self.cuda_graphs[ticker] = cuda_graph_info
                    except Exception as e:
                        logger.warning(f"Could not load CUDA graph for {ticker}: {e}")
                        self.cuda_graphs[ticker] = (None, None, None)
                
                logger.info(f"Loaded model for {ticker}")
                return True
            else:
                logger.warning(f"Model file not found for {ticker}")
                return False
        
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {str(e)}")
            # Print stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_trend_models(self):
        """
        Load all trend prediction models from disk with optimized loading.
        Updated to handle new ensemble format.
        
        Returns:
        --------
        dict
            Dictionary of loaded trend models
        """
        logger.info("Loading trend prediction models...")
        
        models_dir = "models"
        self.trend_models = {}
        self.trend_prediction_modules = {}
        
        if not os.path.exists(models_dir):
            logger.warning("Models directory not found.")
            return {}
        
        # Count loaded models
        n_loaded = 0
        n_failed = 0
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Create a thread pool for parallel loading
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit tasks for each ticker directory
                future_to_ticker = {}
                
                for item in os.listdir(models_dir):
                    item_path = os.path.join(models_dir, item)
                    
                    # Skip ensemble models and non-directories
                    if item.startswith('ensemble_') or not os.path.isdir(item_path):
                        continue
                    
                    # Submit loading task
                    future_to_ticker[executor.submit(self.load_model, item)] = item
                
                # Process results
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        success = future.result()
                        if success:
                            n_loaded += 1
                        else:
                            n_failed += 1
                    except Exception as e:
                        logger.error(f"Error loading model for {ticker}: {str(e)}")
                        n_failed += 1
        
        except (ImportError, AttributeError):
            # Fall back to sequential loading if concurrent.futures not available
            logger.info("Parallel loading not available, using sequential loading")
            
            # Load models for each ticker
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                
                # Skip ensemble models
                if item.startswith('ensemble_'):
                    continue
                    
                # Load individual models
                if os.path.isdir(item_path):
                    ticker = item
                    if self.load_model(ticker):
                        n_loaded += 1
                    else:
                        n_failed += 1
        
        # Load ensemble models with updated approach
        self.ensemble_models = {}
        for item in os.listdir(models_dir):
            if item.startswith('ensemble_') and os.path.isdir(os.path.join(models_dir, item)):
                asset_class = item.replace('ensemble_', '')
                class_dir = os.path.join(models_dir, item)
                
                try:
                    # Load metadata to get tickers and weights
                    metadata_file = os.path.join(class_dir, "metadata.pkl")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'rb') as f:
                            metadata = pickle.load(f)
                        
                        # Get ensemble configuration
                        ensemble_tickers = metadata.get('tickers', [])
                        ensemble_weights = metadata.get('weights', [1/len(ensemble_tickers)] * len(ensemble_tickers))
                        
                        # Store ensemble configuration
                        self.ensemble_models[asset_class] = {
                            'tickers': ensemble_tickers,
                            'weights': ensemble_weights
                        }
                        
                        logger.info(f"Loaded ensemble configuration for {asset_class}")
                except Exception as e:
                    logger.error(f"Error loading ensemble for {asset_class}: {str(e)}")
        
        logger.info(f"Loaded {n_loaded} models, {n_failed} failed")
        logger.info(f"Loaded {len(self.ensemble_models)} ensemble configurations")
        
        return self.trend_models
    
    @torch.no_grad()
    def predict_trend(self, ticker, series=None):
        """
        Predict trend for a single ticker with CUDA optimization.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        series : numpy.ndarray, optional
            Return series data (if None, uses self.returns)
            
        Returns:
        --------
        tuple
            (prediction, direction, strength, attention_weights)
        """
        # Check if model exists
        if ticker not in self.trend_models:
            logger.warning(f"No model found for {ticker}")
            return None, 0, 0, None
        
        model = self.trend_models[ticker]
        
        # Get prediction module
        trend_module = self.trend_prediction_modules.get(ticker)
        
        # Get sequence data
        if series is None:
            if ticker not in self.returns.columns:
                logger.warning(f"No return data found for {ticker}")
                return None, 0, 0, None
            
            series = self.returns[ticker].dropna().values
        
        # Get sequence length
        seq_length = self.model_params.get('sequence_length', 60)
        
        # Check if we have enough data
        if len(series) < seq_length:
            logger.warning(f"Not enough data for {ticker} (need at least {seq_length} points)")
            
            # Try data augmentation if we have some data
            if len(series) > 5:  # Need at least some data to augment
                logger.info(f"Attempting data augmentation for {ticker}")
                augmented_series = augment_time_series(series, augmentation_factor=3)
                
                if len(augmented_series) >= seq_length:
                    logger.info(f"Successfully augmented data for {ticker} from {len(series)} to {len(augmented_series)} points")
                    series = augmented_series
                else:
                    return None, 0, 0, None
            else:
                return None, 0, 0, None
        
        # Scale data if we have a scaler
        if ticker in self.scalers:
            scaler = self.scalers[ticker]
            series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()
        else:
            # Normalize to [-1, 1] range
            series_min, series_max = series.min(), series.max()
            if series_max > series_min:
                series_scaled = -1 + 2 * (series - series_min) / (series_max - series_min)
            else:
                series_scaled = np.zeros_like(series)
        
        # Get the most recent sequence
        last_sequence = series_scaled[-seq_length:]
        
        

        # Prepare input tensor
        if self.cuda_settings.get('enable_channels_last', False) and self.use_cuda:
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(2)
            input_tensor = input_tensor.to(self.device, memory_format=torch.channels_last)
        else:
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # Make prediction using optimized approach
        if trend_module is not None:
            # Check for CUDA Graph
            cuda_graph, static_input, static_outputs = self.cuda_graphs.get(ticker, (None, None, None))
            
            if (cuda_graph is not None and static_input is not None and 
                static_outputs is not None and self.use_cuda):
                # Use CUDA Graph for maximum performance
                
                # Update input data but keep tensor address
                static_input.copy_(input_tensor)
                
                # Execute the graph
                cuda_graph.replay()
                
                # Get results
                pred_tensor, attention_weights = static_outputs
                prediction = pred_tensor.cpu().numpy()[0, 0]
                attention = attention_weights.cpu().numpy()
            else:
                # Use standard prediction with mixed precision
                prediction, attention = trend_module.predict(input_tensor, use_fp16=self.use_mixed_precision)
                prediction = prediction[0, 0]  # Extract scalar value
                attention = attention.cpu().numpy()
        else:
            # Fallback to direct model inference
            model.eval()
            with torch.no_grad():
                if self.use_mixed_precision and self.use_cuda:
                    with autocast():
                        pred_tensor, attention_weights = model(input_tensor)
                else:
                    pred_tensor, attention_weights = model(input_tensor)
                
                prediction = pred_tensor.cpu().numpy()[0, 0]
                attention = attention_weights.cpu().numpy()
        
        # Inverse transform if we have a scaler
        if ticker in self.scalers:
            prediction = self.scalers[ticker].inverse_transform(np.array([[prediction]]))[0][0]
        
        # Calculate direction and strength
        direction = 1 if prediction > 0 else -1 if prediction < 0 else 0
        strength = abs(prediction)
        
        return prediction, direction, strength, attention
    
    def predict_trends(self):
        """
        Generate trend predictions for all assets with CUDA acceleration.
        Now with added ensemble prediction support.
        
        Returns:
        --------
        tuple
            (trend_predictions, trend_direction, trend_strength)
        """
        logger.info("Generating trend predictions with CUDA acceleration...")
        
        # Check if we have models
        if not self.trend_models:
            logger.error("No trend models available. Build or load models first.")
            return None, None, None
        
        # Prepare to store predictions
        predictions = {}
        directions = {}
        strengths = {}
        
        # Get tickers to predict (intersection of models and returns)
        tickers_to_predict = [ticker for ticker in self.trend_models.keys() 
                            if ticker in self.returns.columns]
        
        logger.info(f"Predicting trends for {len(tickers_to_predict)} assets")
        
        # Predict for each individual ticker with error handling
        for ticker in tickers_to_predict:
            try:
                # Get prediction
                prediction, direction, strength, _ = self.predict_trend(ticker)
                
                if prediction is not None:
                    predictions[ticker] = prediction
                    directions[ticker] = direction
                    strengths[ticker] = strength
                    logger.info(f"Generated prediction for {ticker}: {prediction:.4f} (direction: {direction}, strength: {strength:.4f})")
                    
                else:
                    logger.warning(f"Prediction failed for {ticker}, using default values")
                    # Use small non-zero values to avoid scalar issues
                    predictions[ticker] = 0.001
                    directions[ticker] = 1
                    strengths[ticker] = 0.001
                    
            except Exception as e:
                logger.error(f"Error predicting trend for {ticker}: {str(e)}")
                # Use small non-zero values to avoid scalar issues
                predictions[ticker] = 0.001
                directions[ticker] = 1
                strengths[ticker] = 0.001
        
        # Check if we have ensemble models
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            logger.info("Generating ensemble predictions")
            
            # Generate predictions for each asset class using ensembles
            for asset_class, ensemble_config in self.ensemble_models.items():
                try:
                    # Generate ensemble prediction
                    ensemble_prediction, ensemble_direction, ensemble_strength = self.predict_with_ensemble(asset_class)
                    
                    if ensemble_prediction is not None:
                        # Create a virtual ticker for the ensemble
                        ensemble_ticker = f"ENSEMBLE_{asset_class}"
                        
                        # Store ensemble prediction
                        predictions[ensemble_ticker] = ensemble_prediction
                        directions[ensemble_ticker] = ensemble_direction
                        strengths[ensemble_ticker] = ensemble_strength
                        
                        logger.info(f"Added ensemble prediction for {asset_class}")
                        
                        # Also influence individual assets in this class
                        for ticker in self.assets_dict.get(asset_class, []):
                            if ticker in directions:
                                # If direction matches ensemble, boost strength
                                if directions[ticker] == ensemble_direction:
                                    strengths[ticker] *= 1.1  # 10% boost
                                # If opposite, slightly reduce strength
                                elif directions[ticker] == -ensemble_direction and ensemble_direction != 0:
                                    strengths[ticker] *= 0.9  # 10% reduction
                                
                                # Recalculate prediction
                                predictions[ticker] = directions[ticker] * strengths[ticker]
                
                except Exception as e:
                    logger.error(f"Error generating ensemble prediction for {asset_class}: {str(e)}")
        
        # Convert to Series
        self.trend_predictions = pd.Series(predictions)
        self.trend_direction = pd.Series(directions)
        self.trend_strength = pd.Series(strengths)

        # If no predictions were generated, create some default ones
        if self.trend_predictions is None or (isinstance(self.trend_predictions, pd.Series) and self.trend_predictions.empty):
            logger.warning("No model predictions available. Creating default predictions.")
            default_predictions = {}
            default_directions = {}
            default_strengths = {}
            
            # Create more balanced predictions for each asset
            for asset_class, tickers in self.assets_dict.items():
                for i, ticker in enumerate(tickers):
                    # Create more balanced directions with asset class biases
                    if asset_class == 'Equities':
                        direction = 1 if i % 3 != 0 else -1  # Mostly bullish for equities
                    elif asset_class == 'Bonds':
                        direction = 1 if i % 3 != 0 else -1  # Mostly bullish for bonds
                    elif asset_class == 'Commodities':
                        direction = 1 if i % 2 == 0 else -1  # Balanced for commodities
                    else:
                        direction = -1 if i % 2 == 0 else 1  # Balanced for currencies
                    
                    # More realistic strength values (lower)
                    strength = 0.05 + (0.03 * (i % 5))  # Range from 0.05 to 0.17
                    
                    default_predictions[ticker] = direction * strength
                    default_directions[ticker] = direction
                    default_strengths[ticker] = strength
            
            # Store the default predictions
            self.trend_predictions = pd.Series(default_predictions)
            self.trend_direction = pd.Series(default_directions)
            self.trend_strength = pd.Series(default_strengths)
        
            logger.info(f"Generated predictions for {len(predictions)} assets")
        else:
            # Convert to Series
            self.trend_predictions = pd.Series(predictions)
            self.trend_direction = pd.Series(directions)
            self.trend_strength = pd.Series(strengths)
            
            logger.info(f"Generated {len(predictions)} asset predictions from models")
        return self.trend_predictions, self.trend_direction, self.trend_strength
    
    def adjust_for_economic_uncertainty(self):
        """
        Adjust trend signals based on economic uncertainty for better risk management.
        Now with ensemble prediction integration.
        
        Returns:
        --------
        tuple
            (adjusted_direction, adjusted_strength)
        """
        logger.info("Adjusting for economic uncertainty...")
        
        # Check if we have predictions
        if self.trend_direction is None or self.trend_strength is None:
            logger.error("No trend predictions available. Run predict_trends() first.")
            return None, None
        
        # Get economic data
        if not hasattr(self, 'economic_data') or self.economic_data is None or self.economic_data.empty:
            logger.warning("No economic data available. Skipping adjustment.")
            return self.trend_direction, self.trend_strength
        
        # Get the most recent economic data
        latest_econ = self.economic_data.iloc[-1]
        
        # Extract economic indicators with better error handling
        try:
            treasury_yield = next((latest_econ.get(col) for col in latest_econ.index 
                                if 'treasury_yield' in col.lower() and '10year' in col.lower()), 3.0)
        except:
            treasury_yield = 3.0  # Default if not found
        
        try:
            inflation = next((latest_econ.get(col) for col in latest_econ.index 
                            if 'inflation' in col.lower() or 'cpi' in col.lower()), 3.0)
        except:
            inflation = 3.0  # Default if not found
        
        try:
            unemployment = next((latest_econ.get(col) for col in latest_econ.index 
                            if 'unemployment' in col.lower()), 4.0)
        except:
            unemployment = 4.0  # Default if not found
        
        try:
            gdp_growth = next((latest_econ.get(col) for col in latest_econ.index 
                            if 'gdp' in col.lower() and 'real' in col.lower()), 2.0)
        except:
            gdp_growth = 2.0  # Default if not found
        
        # Calculate economic regime
        # Sophisticated regime classification: expansion, slowdown, recession, recovery
        expansion = gdp_growth > 2.5 and inflation < 4.0 and unemployment < 5.0
        recession = gdp_growth < 0.5 or unemployment > 6.0
        recovery = gdp_growth > 1.0 and gdp_growth < 2.5 and unemployment > 5.0
        slowdown = not (expansion or recession or recovery)
        
        # Calculate uncertainty score based on economic variables and volatility
        uncertainty = (10 * treasury_yield + 5 * inflation + 5 * unemployment - 2 * gdp_growth) / 3
        uncertainty = min(100, max(0, uncertainty))  # Cap between 0-100
        
        logger.info(f"Economic regime analysis:")
        if expansion:
            logger.info("  Detected economic EXPANSION")
            regime = "EXPANSION"
        elif recession:
            logger.info("  Detected economic RECESSION")
            regime = "RECESSION"
        elif recovery:
            logger.info("  Detected economic RECOVERY")
            regime = "RECOVERY"
        elif slowdown:
            logger.info("  Detected economic SLOWDOWN")
            regime = "SLOWDOWN"
        else:
            regime = "UNKNOWN"
        logger.info(f"  Uncertainty score: {uncertainty:.1f}/100")
        
        # Copy directions and strengths for adjustment
        adjusted_direction = self.trend_direction.copy()
        adjusted_strength = self.trend_strength.copy()
        
        # Identify ensemble predictions
        ensemble_predictions = {
            ticker: (adjusted_direction[ticker], adjusted_strength[ticker])
            for ticker in adjusted_direction.index
            if ticker.startswith('ENSEMBLE_')
        }
        
        # Apply regime-specific adjustments
        for asset_class, tickers in self.assets_dict.items():
            # Check if we have an ensemble for this asset class
            ensemble_ticker = f"ENSEMBLE_{asset_class}"
            has_ensemble = ensemble_ticker in ensemble_predictions
            
            # Get ensemble direction and strength if available
            ensemble_direction = ensemble_predictions.get(ensemble_ticker, (0, 0))[0]
            ensemble_strength = ensemble_predictions.get(ensemble_ticker, (0, 0))[1]
            
            for ticker in tickers:
                if ticker in adjusted_strength.index:
                    # First, consider ensemble view if available
                    if has_ensemble and ensemble_direction != 0:
                        # If ticker direction matches ensemble, boost strength
                        if adjusted_direction[ticker] == ensemble_direction:
                            adjusted_strength[ticker] *= 1.15  # 15% boost
                        # If ticker contradicts strong ensemble, reduce strength
                        elif adjusted_direction[ticker] * ensemble_direction < 0 and ensemble_strength > 0.1:
                            adjusted_strength[ticker] *= 0.85  # 15% reduction
                    
                    # Then apply regime-specific adjustments
                    if asset_class == 'Commodities':
                        if expansion:
                            # In expansion, commodities do well with rising demand
                            if adjusted_direction[ticker] > 0:
                                adjusted_strength[ticker] *= 1.1
                        elif recession:
                            # In recession, commodities tend to weaken
                            if adjusted_direction[ticker] < 0:
                                adjusted_strength[ticker] *= 1.2
                        elif recovery:
                            # In recovery, industrial commodities do well
                            if ticker in ['COPPER', 'ALUMINUM', 'OIL'] and adjusted_direction[ticker] > 0:
                                adjusted_strength[ticker] *= 1.15
                    
                    elif asset_class == 'Currencies':
                        # Adjust based on relative monetary policy
                        if expansion and uncertainty > 60:
                            # In expansion with high uncertainty, safe haven currencies strengthen
                            if ticker in ['JPY', 'CHF'] and adjusted_direction[ticker] > 0:
                                adjusted_strength[ticker] *= 1.1
                        elif recession:
                            # In recession, USD typically strengthens
                            if ticker not in ['USD'] and adjusted_direction[ticker] < 0:
                                adjusted_strength[ticker] *= 1.1
                    
                    elif asset_class == 'Bonds':
                        if recession or slowdown:
                            # In slowdown or recession, bonds typically rally
                            if ensemble_direction != 0:
                                # Use ensemble direction if available
                                adjusted_direction[ticker] = ensemble_direction
                            else:
                                adjusted_direction[ticker] = 1
                            adjusted_strength[ticker] *= 1.2
                        elif expansion and inflation > 3.5:
                            # In expansion with high inflation, bonds may weaken
                            if ensemble_direction != 0:
                                # Use ensemble direction if available
                                adjusted_direction[ticker] = ensemble_direction
                            else:
                                adjusted_direction[ticker] = -1
                            adjusted_strength[ticker] *= 1.1
                    
                    elif asset_class == 'Equities':
                        if recession:
                            # In recession, equity trends are likely downward
                            if adjusted_direction[ticker] < 0:
                                adjusted_strength[ticker] *= 1.15
                            elif adjusted_direction[ticker] > 0:
                                # Reduce strength of bullish signals
                                adjusted_strength[ticker] *= 0.7
                        elif recovery:
                            # In recovery, equity trends are likely upward
                            if adjusted_direction[ticker] > 0:
                                adjusted_strength[ticker] *= 1.1
        
        # Uncertainty-based global adjustments
        if uncertainty > 70:
            logger.info("High uncertainty detected - applying defensive adjustments")
            
            # Increase strength of negative equity signals
            for ticker in self.assets_dict.get('Equities', []):
                if ticker in adjusted_direction.index and adjusted_direction[ticker] < 0:
                    adjusted_strength[ticker] *= 1.2
            
            # Favor bonds in high uncertainty
            for ticker in self.assets_dict.get('Bonds', []):
                if ticker in adjusted_direction.index:
                    adjusted_direction[ticker] = 1
                    adjusted_strength[ticker] *= 1.15
            
            # Favor safe haven currencies
            for ticker in ['JPY', 'CHF']:
                if ticker in adjusted_direction.index:
                    adjusted_direction[ticker] = 1
                    adjusted_strength[ticker] *= 1.1
        
        # Ensure balance between long and short positions
        # Exclude ensemble tickers from this analysis
        standard_tickers = [t for t in adjusted_direction.index if not t.startswith('ENSEMBLE_')]
        
        long_count = sum(1 for t in standard_tickers if adjusted_direction[t] > 0)
        short_count = sum(1 for t in standard_tickers if adjusted_direction[t] < 0)
        total_count = len(standard_tickers)
        
        # Target at least 30% in both directions
        min_target = max(int(total_count * 0.3), 3)
        
        if long_count < min_target and short_count > min_target:
            logger.info(f"Too few long positions ({long_count}/{total_count}), adding more for balance")
            
            # Find short positions to flip (weakest signals first)
            short_strengths = pd.Series({
                k: v for k, v in adjusted_strength.items() 
                if k in standard_tickers and adjusted_direction[k] < 0
            })
            
            # Sort by increasing strength
            to_flip = short_strengths.nsmallest(min_target - long_count).index
            
            # Flip direction
            for ticker in to_flip:
                adjusted_direction[ticker] = 1
                adjusted_strength[ticker] *= 0.7  # Reduce strength when flipping
                logger.info(f"  Flipped {ticker} to positive (balancing adjustment)")
        
        elif short_count < min_target and long_count > min_target:
            logger.info(f"Too few short positions ({short_count}/{total_count}), adding more for balance")
            
            # Find long positions to flip (weakest signals first)
            long_strengths = pd.Series({
                k: v for k, v in adjusted_strength.items() 
                if k in standard_tickers and adjusted_direction[k] > 0
            })
            
            # Sort by increasing strength
            to_flip = long_strengths.nsmallest(min_target - short_count).index
            
            # Flip direction
            for ticker in to_flip:
                adjusted_direction[ticker] = -1
                adjusted_strength[ticker] *= 0.7  # Reduce strength when flipping
                logger.info(f"  Flipped {ticker} to negative (balancing adjustment)")
        
        # Store adjusted predictions
        self.adjusted_direction = adjusted_direction
        self.adjusted_strength = adjusted_strength
        
        # Store economic regime for later reference
        self.economic_regime = regime
        self.uncertainty_score = uncertainty
        
        logger.info("Adjusted trend predictions for economic regime")
        return self.adjusted_direction, self.adjusted_strength
    
    def adjust_for_news_sentiment(self):
        """
        Further adjust trend signals based on news sentiment for market reaction.
        
        Returns:
        --------
        tuple
            (final_direction, final_strength)
        """
        logger.info("Adjusting for news sentiment...")
        
        # Check if we have adjusted predictions
        if not hasattr(self, 'adjusted_direction') or not hasattr(self, 'adjusted_strength'):
            logger.warning("No adjusted predictions available. Run adjust_for_economic_uncertainty() first.")
            return self.trend_direction, self.trend_strength
        
        # Load news sentiment if needed
        if not hasattr(self, 'news_sentiment') or self.news_sentiment is None or self.news_sentiment.empty:
            if self.av_client:
                self.load_news_sentiment()
            else:
                logger.warning("No news sentiment data available and no API key to load it.")
                return self.adjusted_direction, self.adjusted_strength
        
        # Check if we have news sentiment data
        if self.news_sentiment.empty:
            logger.warning("No news sentiment data available. Skipping adjustment.")
            return self.adjusted_direction, self.adjusted_strength
        
        # Start with the economically adjusted values
        final_direction = self.adjusted_direction.copy()
        final_strength = self.adjusted_strength.copy()
        
        # Define sentiment thresholds
        STRONG_POSITIVE = 0.5
        POSITIVE = 0.2
        NEUTRAL_UPPER = 0.2
        NEUTRAL_LOWER = -0.2
        NEGATIVE = -0.2
        STRONG_NEGATIVE = -0.5
        
        # 1. Calculate overall market sentiment
        overall_sentiment = self.news_sentiment['ticker_sentiment_score'].mean()
        overall_sentiment_desc = "neutral"
        
        if overall_sentiment > STRONG_POSITIVE:
            overall_sentiment_desc = "strongly positive"
        elif overall_sentiment > POSITIVE:
            overall_sentiment_desc = "positive"
        elif overall_sentiment < STRONG_NEGATIVE:
            overall_sentiment_desc = "strongly negative"
        elif overall_sentiment < NEGATIVE:
            overall_sentiment_desc = "negative"
        
        logger.info(f"Overall market sentiment: {overall_sentiment:.2f} ({overall_sentiment_desc})")
        
        # 2. Calculate sentiment by asset class
        asset_class_sentiment = {}
        
        for asset_class, tickers in self.assets_dict.items():
            # Get sentiment for tickers in this asset class
            class_sentiment = []
            
            for ticker in tickers:
                ticker_sentiment = self.news_sentiment[self.news_sentiment['ticker'] == ticker]
                
                if not ticker_sentiment.empty:
                    avg_sentiment = ticker_sentiment['ticker_sentiment_score'].mean()
                    relevance = ticker_sentiment['relevance_score'].mean()
                    
                    # Weight by relevance
                    weighted_sentiment = avg_sentiment * relevance
                    class_sentiment.append(weighted_sentiment)
            
            # Calculate average sentiment for this asset class
            if class_sentiment:
                asset_class_sentiment[asset_class] = sum(class_sentiment) / len(class_sentiment)
                logger.info(f"  {asset_class} sentiment: {asset_class_sentiment[asset_class]:.2f}")
            else:
                asset_class_sentiment[asset_class] = 0
        
        # 3. Apply sentiment-based adjustments
        
        # 3.1. Overall market sentiment adjustments
        if overall_sentiment > POSITIVE:
            # Positive market sentiment: strengthen upward trends
            for ticker, direction in final_direction.items():
                if direction > 0:  # Long position
                    # Scale factor based on sentiment strength
                    factor = 1.0 + (0.15 * (overall_sentiment - POSITIVE) / (1 - POSITIVE))
                    final_strength[ticker] *= factor
            
            # Reduce emphasis on safe haven assets
            safe_havens = ['TREASURY_YIELD', 'JPY', 'CHF', 'GOLD']
            for ticker in safe_havens:
                if ticker in final_strength.index:
                    final_strength[ticker] *= 0.9
        
        elif overall_sentiment < NEGATIVE:
            # Negative market sentiment: strengthen downward trends
            for ticker, direction in final_direction.items():
                if direction < 0:  # Short position
                    # Scale factor based on sentiment strength
                    factor = 1.0 + (0.15 * (NEGATIVE - overall_sentiment) / (NEGATIVE - (-1)))
                    final_strength[ticker] *= factor
            
            # Strengthen safe haven assets
            safe_havens = ['TREASURY_YIELD', 'JPY', 'CHF', 'GOLD']
            for ticker in safe_havens:
                if ticker in final_strength.index:
                    if ticker == 'TREASURY_YIELD':
                        final_direction[ticker] = -1  # Bonds up = yields down
                    else:
                        final_direction[ticker] = 1
                    final_strength[ticker] *= 1.1
        
        # 3.2. Asset class sentiment adjustments
        for asset_class, sentiment in asset_class_sentiment.items():
            # Skip neutral sentiment
            if NEUTRAL_LOWER <= sentiment <= NEUTRAL_UPPER:
                continue
            
            tickers = self.assets_dict.get(asset_class, [])
            
            if sentiment > POSITIVE:
                # Positive asset class sentiment
                for ticker in tickers:
                    if ticker in final_direction.index:
                        # Strengthen upward trends or possibly flip weak downward trends
                        if final_direction[ticker] > 0:
                            # Strengthen existing positive trend
                            factor = 1.0 + (0.1 * (sentiment - POSITIVE) / (1 - POSITIVE))
                            final_strength[ticker] *= factor
                        elif final_strength[ticker] < 0.1:
                            # Flip weak negative trend
                            final_direction[ticker] = 1
                            final_strength[ticker] *= 0.7  # Reduce strength when flipping
                            logger.info(f"  Flipped {ticker} to positive based on asset class sentiment")
            
            elif sentiment < NEGATIVE:
                # Negative asset class sentiment
                for ticker in tickers:
                    if ticker in final_direction.index:
                        # Strengthen downward trends or possibly flip weak upward trends
                        if final_direction[ticker] < 0:
                            # Strengthen existing negative trend
                            factor = 1.0 + (0.1 * (NEGATIVE - sentiment) / (NEGATIVE - (-1)))
                            final_strength[ticker] *= factor
                        elif final_strength[ticker] < 0.1:
                            # Flip weak positive trend
                            final_direction[ticker] = -1
                            final_strength[ticker] *= 0.7  # Reduce strength when flipping
                            logger.info(f"  Flipped {ticker} to negative based on asset class sentiment")
        
        # 3.3. Ticker-specific sentiment adjustments
        for ticker in final_direction.index:
            # Get sentiment specifically for this ticker
            ticker_specific = self.news_sentiment[self.news_sentiment['ticker'] == ticker]
            
            if not ticker_specific.empty:
                specific_sentiment = ticker_specific['ticker_sentiment_score'].mean()
                specific_relevance = ticker_specific['relevance_score'].mean()
                
                # Only apply if sentiment is strong and relevance is high
                if abs(specific_sentiment) > 0.3 and specific_relevance > 0.7:
                    if specific_sentiment > STRONG_POSITIVE:
                        # Strong positive sentiment for this ticker
                        final_direction[ticker] = 1
                        final_strength[ticker] *= 1.3
                        logger.info(f"  Strong positive sentiment for {ticker} (score: {specific_sentiment:.2f})")
                    elif specific_sentiment < STRONG_NEGATIVE:
                        # Strong negative sentiment for this ticker
                        final_direction[ticker] = -1
                        final_strength[ticker] *= 1.3
                        logger.info(f"  Strong negative sentiment for {ticker} (score: {specific_sentiment:.2f})")
        
        # Store final adjusted predictions
        self.final_direction = final_direction
        self.final_strength = final_strength
        
        logger.info("Final trend predictions adjusted for news sentiment")
        return self.final_direction, self.final_strength
    
    def export_to_csv(self, filename='etf_allocation.csv'):
        """
        Export predictions and allocations to CSV.
        
        Parameters:
        -----------
        filename : str, optional
            Name of CSV file
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions and allocations
        """
        # Check if we have final predictions
        if not hasattr(self, 'final_direction') or not hasattr(self, 'final_strength'):
            logger.warning("No final predictions available. Run adjust_for_news_sentiment() first.")
            if not hasattr(self, 'trend_direction') or not hasattr(self, 'trend_strength'):
                logger.warning("No trend predictions available either. Run predict_trends() first.")
                return None
            
            # Use trend predictions if no final predictions
            direction = self.trend_direction
            strength = self.trend_strength
        else:
            direction = self.final_direction
            strength = self.final_strength
        
        # Create DataFrame
        export_df = pd.DataFrame({
            'Direction': direction,
            'Strength': strength
        })
        
        # Add asset class information
        export_df['Asset_Class'] = 'Unknown'
        
        for asset_class, tickers in self.assets_dict.items():
            for ticker in tickers:
                if ticker in export_df.index:
                    export_df.loc[ticker, 'Asset_Class'] = asset_class
        
        # Add volatility if available
        if self.returns is not None:
            volatilities = self.returns.ewm(alpha=0.06).std() * np.sqrt(252)  # 30-day EWM, annualized
            export_df['Volatility'] = volatilities.iloc[-1]
        else:
            export_df['Volatility'] = 0.20  # Default
        
        # Save to CSV
        export_df.to_csv(filename)
        logger.info(f"Exported predictions to {filename}")
        
        return export_df
    
    def get_attention_weights(self, top_n=5):
        """
        Get attention weights for visualization.
        
        Parameters:
        -----------
        top_n : int
            Number of top assets to include
            
        Returns:
        --------
        dict
            Dictionary of attention weights for top assets
        """
        attention_weights = {}
        
        # Check if we have final predictions
        if not hasattr(self, 'final_direction') or not hasattr(self, 'final_strength'):
            # Use trend predictions if no final predictions
            if not hasattr(self, 'trend_direction') or not hasattr(self, 'trend_strength'):
                logger.warning("No predictions available.")
                return attention_weights
            
            direction = self.trend_direction
            strength = self.trend_strength
        else:
            direction = self.final_direction
            strength = self.final_strength
        
        # Calculate absolute signal strength
        signal_strength = direction.abs() * strength
        
        # Get top N assets by signal strength
        top_assets = signal_strength.nlargest(top_n).index
        
        # Get attention weights for top assets
        for ticker in top_assets:
            try:
                # Predict again to get attention weights
                _, _, _, attention = self.predict_trend(ticker)
                if attention is not None:
                    attention_weights[ticker] = attention
            except Exception as e:
                logger.error(f"Error getting attention weights for {ticker}: {str(e)}")
        
        return attention_weights
    
    def benchmark_performance(self, sample_size=10):
        """
        Benchmark CUDA performance across different optimization techniques.
        
        Parameters:
        -----------
        sample_size : int
            Number of models to include in benchmark
            
        Returns:
        --------
        dict
            Performance results
        """
        if not self.trend_models:
            logger.error("No models available for benchmarking.")
            return None
        
        # Select a random sample of models
        import random
        tickers = list(self.trend_models.keys())
        if len(tickers) > sample_size:
            benchmark_tickers = random.sample(tickers, sample_size)
        else:
            benchmark_tickers = tickers
        
        logger.info(f"Running performance benchmark with {len(benchmark_tickers)} models...")
        
        # Initialize results dictionary
        results = {}
        
        # Get sequence length
        seq_length = self.model_params.get('sequence_length', 60)
        
        # Create sample input
        sample_input = torch.zeros((1, seq_length, 1), device=self.device)
        
        # Run benchmarks for each selected model
        for ticker in benchmark_tickers:
            try:
                # Get prediction module
                model_module = self.trend_prediction_modules.get(ticker)
                
                if model_module is not None:
                    # Run benchmarks
                    ticker_results = model_module.benchmark_inference(sample_input, iterations=100)
                    results[ticker] = ticker_results
                    
                    # Log results
                    logger.info(f"Benchmark for {ticker}:")
                    for config, time_ms in ticker_results.items():
                        if not config.endswith('_speedup'):
                            logger.info(f"  {config:20s}: {time_ms:.3f} ms")
            except Exception as e:
                logger.error(f"Error benchmarking {ticker}: {str(e)}")
        
        # Calculate aggregate statistics
        if results:
            # Initialize aggregates
            aggregates = {}
            
            # Process each config type
            for ticker, ticker_results in results.items():
                for config, value in ticker_results.items():
                    if config not in aggregates:
                        aggregates[config] = []
                    aggregates[config].append(value)
            
            # Calculate statistics
            stats = {}
            for config, values in aggregates.items():
                stats[config] = {
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values)
                }
            
            # Log aggregate stats
            logger.info("Aggregate benchmark results:")
            for config, stat in stats.items():
                if not config.endswith('_speedup'):
                    logger.info(f"  {config:20s}: {stat['mean']:.3f} ms (±{stat['std']:.3f} ms)")
            
            # Add stats to results
            results['aggregate_stats'] = stats
        
        return results
    
