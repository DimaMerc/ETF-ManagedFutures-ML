# feature_engineering_integration.py
"""
Integration module for advanced feature engineering with CudaOptimizedTrendPredictor.
Enhances the existing predictor with specialized features for commodities, FX pairs, and bonds.
"""

import os
import numpy as np
import pandas as pd
import logging
from advanced_features import (
    generate_features_for_asset,
    normalize_features_for_lstm,
    select_top_features,
    create_feature_pipeline
)
from alpha_vantage_client_optimized import CommodityData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineeringEnhancer:
    """
    Enhanced feature engineering for CudaOptimizedTrendPredictor.
    Integrates advanced features for different asset classes.
    """
    
    def __init__(self, predictor, cache_dir="feature_cache", n_features=20, normalize_method="asset_class"):
        """
        Initialize the feature engineering enhancer.
        
        Parameters:
        -----------
        predictor : CudaOptimizedTrendPredictor
            The LSTM predictor to enhance
        cache_dir : str
            Directory to cache engineered features
        n_features : int
            Number of top features to select for each asset
        normalize_method : str
            Method for normalizing features ('asset_class', 'volatility', 'robust', 'standard', 'minmax')
        """
        self.predictor = predictor
        self.cache_dir = cache_dir
        self.n_features = n_features
        self.normalize_method = normalize_method
        
        # Dictionary to store engineered features
        self.asset_features = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Feature engineering enhancer initialized with {n_features} features per asset")
    
    def enhance_predictor(self):
        """
        Enhance the predictor with advanced features for all asset classes.
        This is the main integration method that sets up the predictor with new features.
        
        Returns:
        --------
        bool
            True if enhancement was successful
        """
        logger.info("Enhancing predictor with advanced features...")
        
        # Check if predictor has returns data
        if self.predictor.returns is None or self.predictor.returns.empty:
            logger.error("Predictor has no returns data. Load data first.")
            return False
        
        # Check if predictor has assets dictionary
        if not hasattr(self.predictor, 'assets_dict') or not self.predictor.assets_dict:
            logger.error("Predictor has no assets dictionary.")
            return False
        
        # Create asset-to-class mapping
        asset_class_map = {}
        for asset_class, tickers in self.predictor.assets_dict.items():
            for ticker in tickers:
                asset_class_map[ticker] = asset_class
        
        # Prepare data dictionary for advanced feature generation
        price_data = {}
        returns_data = {}
        
        # Extract prices from returns (reconstruct from cumulative returns)
        try:
            for ticker in self.predictor.returns.columns:
                # Start with a base price of 100
                base_price = 100
                returns = self.predictor.returns[ticker].fillna(0)
                prices = (1 + returns).cumprod() * base_price
                
                price_data[ticker] = prices
                returns_data[ticker] = returns
                
                logger.info(f"Prepared data for {ticker}")
        except Exception as e:
            logger.error(f"Error preparing price data: {str(e)}")
            return False
        
        # Generate advanced features for each asset
        successfully_enhanced = []
        failed_enhancements = []
        
        for ticker, asset_class in asset_class_map.items():
            if ticker in price_data:
                try:
                    # Check if features are cached
                    cache_file = os.path.join(self.cache_dir, f"{ticker}_features.pkl")
                    
                    if os.path.exists(cache_file):
                        # Load from cache
                        try:
                            logger.info(f"Loading cached features for {ticker}")
                            features = pd.read_pickle(cache_file)
                            self.asset_features[ticker] = features
                            successfully_enhanced.append(ticker)
                            continue
                        except Exception as e:
                            logger.warning(f"Error loading cached features for {ticker}: {str(e)}")
                    
                    # Generate appropriate features based on asset class
                    logger.info(f"Generating {asset_class} features for {ticker}")
                    
                    # Ensure prices is a simple Series, not a DataFrame
                    prices = price_data[ticker]
                    if isinstance(prices, pd.DataFrame):
                        # Handle DataFrame case - extract relevant series
                        if 'close' in prices.columns:
                            prices = prices['close']
                        elif 'adjusted_close' in prices.columns:
                            prices = prices['adjusted_close']
                        elif 'value' in prices.columns:
                            prices = prices['value']
                        else:
                            # Get first numeric column
                            numeric_cols = prices.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                prices = prices[numeric_cols[0]]
                            else:
                                prices = prices.iloc[:, 0]  # Fall back to first column
                    
                    # Ensure returns is a simple Series too
                    returns = returns_data[ticker]
                    if isinstance(returns, pd.DataFrame):
                        # Handle DataFrame case - extract relevant series
                        if 'returns' in returns.columns:
                            returns = returns['returns']
                        else:
                            numeric_cols = returns.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                returns = returns[numeric_cols[0]]
                            else:
                                returns = returns.iloc[:, 0]  # Fall back to first column
                    
                    # Convert to numeric Series with proper index
                    prices = pd.to_numeric(prices, errors='coerce')
                    returns = pd.to_numeric(returns, errors='coerce')
                    
                    features = create_feature_pipeline(
                        prices=prices,
                        asset_class=asset_class,
                        normalize=False,  # Will normalize across assets later
                        select_features=False,  # Will select after generating all features
                        n_features=self.n_features
                    )
                    
                    # Store raw features
                    self.asset_features[ticker] = features
                    
                    # Cache features
                    try:
                        features.to_pickle(cache_file)
                    except Exception as e:
                        logger.warning(f"Error caching features for {ticker}: {str(e)}")
                    
                    successfully_enhanced.append(ticker)
                    
                except Exception as e:
                    logger.error(f"Error generating features for {ticker}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    failed_enhancements.append(ticker)
            else:
                logger.warning(f"No price data available for {ticker}")
                failed_enhancements.append(ticker)


        # After generating basic features, add return-focused features
        for ticker, features in self.asset_features.items():
            asset_class = asset_class_map.get(ticker, "Equities")
            
            try:
                # Add return-predictive features
                enhanced_features = add_return_prediction_features(features, asset_class)
                self.asset_features[ticker] = enhanced_features
                logger.info(f"Added return-predictive features for {ticker}")
            except Exception as e:
                logger.error(f"Error adding return-predictive features for {ticker}: {str(e)}")
        
        # Log enhancement results
        logger.info(f"Successfully enhanced {len(successfully_enhanced)} assets.")
        if failed_enhancements:
            logger.warning(f"Failed to enhance {len(failed_enhancements)} assets: {failed_enhancements}")
        
        # Apply feature normalization across all assets
        if successfully_enhanced:
            try:
                self._normalize_features(asset_class_map)
            except Exception as e:
                logger.error(f"Error normalizing features: {str(e)}")
        
        # Apply feature selection across all assets
        try:
            self._select_top_features(returns_data)
        except Exception as e:
            logger.error(f"Error selecting top features: {str(e)}")
        
        # Extend predictor with new features
        self._extend_predictor()
        
        return len(successfully_enhanced) > 0
    
    def _normalize_features(self, asset_class_map):
        """
        Apply normalization across all assets.
        
        Parameters:
        -----------
        asset_class_map : dict
            Mapping from asset tickers to asset classes
        
        Returns:
        --------
        None
        """
        logger.info(f"Normalizing features using method: {self.normalize_method}")
        
        if self.normalize_method == "asset_class":
            # Apply different normalization methods based on asset class
            for ticker, features in self.asset_features.items():
                asset_class = asset_class_map.get(ticker, "Equities")
                
                if asset_class == "Commodities":
                    # Use robust scaling for commodities (fat tails)
                    self.asset_features[ticker] = normalize_features_for_lstm(features, method="robust")
                    
                elif asset_class == "Currencies":
                    # Use standard scaling for FX
                    self.asset_features[ticker] = normalize_features_for_lstm(features, method="standard")
                    
                elif asset_class == "Bonds":
                    # Use minmax scaling for bonds
                    self.asset_features[ticker] = normalize_features_for_lstm(features, method="minmax")
                    
                else:  # Equities
                    # Use quantile transform for equities
                    self.asset_features[ticker] = normalize_features_for_lstm(features, method="quantile")
                    
                logger.info(f"Normalized features for {ticker} using {asset_class}-specific method")
                
        elif self.normalize_method == "volatility":
            # Normalize by asset volatility
            returns = self.predictor.returns
            
            for ticker, features in self.asset_features.items():
                if ticker in returns:
                    # Calculate annualized volatility
                    volatility = returns[ticker].std() * np.sqrt(252)
                    
                    # Skip if volatility is 0 or NaN
                    if volatility <= 0 or np.isnan(volatility):
                        continue
                    
                    # Normalize features by dividing by volatility
                    normalized = features.copy()
                    for col in normalized.columns:
                        normalized[col] = normalized[col] / volatility
                    
                    self.asset_features[ticker] = normalized
                    logger.info(f"Normalized features for {ticker} by volatility")
                    
        else:
            # Apply the same normalization method to all assets
            for ticker, features in self.asset_features.items():
                self.asset_features[ticker] = normalize_features_for_lstm(
                    features, method=self.normalize_method
                )
                logger.info(f"Normalized features for {ticker} using {self.normalize_method} method")
    
    def _select_top_features(self, returns_data, method="correlation"):
        """
        Select top features for each asset based on correlation with returns.
        
        Parameters:
        -----------
        returns_data : dict
            Dictionary of return series for each asset
        method : str
            Feature selection method
            
        Returns:
        --------
        None
        """
        logger.info(f"Selecting top {self.n_features} features using {method} method")
        
        for ticker, features in self.asset_features.items():
            if ticker in returns_data and not features.empty:
                try:
                    # Select top features
                    top_features = select_top_features(
                        features_df=features,
                        returns=returns_data[ticker],
                        n_features=min(self.n_features, len(features.columns)),
                        method=method
                    )
                    
                    # Update features
                    self.asset_features[ticker] = top_features
                    logger.info(f"Selected top {len(top_features.columns)} features for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error selecting features for {ticker}: {str(e)}")
    
    def _extend_predictor(self):
        """
        Extend the trend predictor with engineered features.
        
        Returns:
        --------
        None
        """
        logger.info("Extending predictor with engineered features")
        
        # Store features in predictor
        if not hasattr(self.predictor, 'engineered_features'):
            self.predictor.engineered_features = {}
            
        self.predictor.engineered_features = self.asset_features
        
        # Integrate feature engineering with predictor prepare_lstm_data method
        original_prepare_lstm_data = self.predictor.prepare_lstm_data
        
        def enhanced_prepare_lstm_data(self, series, seq_length, augment_if_needed=True):
            """
            Enhanced version of prepare_lstm_data that incorporates engineered features.
            This method extends the original to use engineered features when available.
            
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
            # Determine ticker for this series
            ticker = None
            
            if hasattr(self, 'current_ticker'):
                ticker = self.current_ticker
                
            # If we have engineered features for this ticker, use them
            if ticker and hasattr(self, 'engineered_features') and ticker in self.engineered_features:
                try:
                    # Get engineered features
                    features = self.engineered_features[ticker]
                    
                    # Align with series length
                    if len(features) != len(series):
                        logger.warning(f"Feature length ({len(features)}) doesn't match series length ({len(series)})")
                        # Truncate to the shorter length
                        min_length = min(len(features), len(series))
                        features = features.iloc[-min_length:]
                        series = series[-min_length:]
                    
                    # Create sequences with multiple features (original plus engineered)
                    X, y = [], []
                    for i in range(len(features) - seq_length):
                        # Extract feature sequence
                        feature_seq = features.iloc[i:i+seq_length].values
                        
                        # Target is the next value in the original series
                        target = series[i+seq_length]
                        
                        X.append(feature_seq)
                        y.append(target)
                    
                    # Convert to numpy arrays
                    if X and y:
                        X_np = np.array(X)
                        y_np = np.array(y)
                        
                        # Convert to tensors and move to device (from original method)
                        import torch
                        X_tensor = torch.tensor(X_np, dtype=torch.float32)
                        y_tensor = torch.tensor(y_np, dtype=torch.float32)
                        
                        # Move to device
                        X_tensor = X_tensor.to(self.device)
                        y_tensor = y_tensor.to(self.device)
                        
                        return X_tensor, y_tensor
                        
                except Exception as e:
                    logger.error(f"Error using engineered features: {str(e)}")
                    # Fall back to original method
            
            # Call original method if engineered features can't be used
            return original_prepare_lstm_data(self, series, seq_length, augment_if_needed)
        
        # Replace the method with enhanced version
        self.predictor.prepare_lstm_data = enhanced_prepare_lstm_data.__get__(self.predictor, type(self.predictor))
        
        # Patch predict_trend to set current ticker
        original_predict_trend = self.predictor.predict_trend
        
        def enhanced_predict_trend(self, ticker, series=None):
            """
            Enhanced version of predict_trend that sets current ticker for feature lookup.
            
            Parameters:
            -----------
            ticker : str
                Ticker symbol
            series : numpy.ndarray, optional
                Return series data
                
            Returns:
            --------
            tuple
                (prediction, direction, strength, attention_weights)
            """
            # Set current ticker for feature lookup
            self.current_ticker = ticker
            
            # Call original method
            result = original_predict_trend(self, ticker, series)
            
            # Clear current ticker
            self.current_ticker = None
            
            return result
        
        # Replace the method with enhanced version
        self.predictor.predict_trend = enhanced_predict_trend.__get__(self.predictor, type(self.predictor))
        
        logger.info("Predictor extended with engineered features")

    

    

def enhance_technical_indicators(av_client, tickers, asset_classes):
    """
    Generate technical indicators for assets where Alpha Vantage doesn't provide them.
    Serves as a replacement for the Alpha Vantage technical indicator API.
    
    Parameters:
    -----------
    av_client : AlphaVantageClient
        Alpha Vantage client instance
    tickers : list
        List of ticker symbols
    asset_classes : dict
        Dictionary mapping tickers to asset classes
        
    Returns:
    --------
    dict
        Dictionary mapping tickers to technical indicators
    """
    from advanced_features import (
        calculate_commodity_features,
        calculate_fx_features,
        calculate_bond_features,
        calculate_statistical_moments,
        calculate_returns_features,
        calculate_volatility_features,
        calculate_time_series_features
    )
    
    indicators = {}
    
    # Process each ticker
    for ticker in tickers:
        asset_class = asset_classes.get(ticker, "Equities")
        
        try:
            # Get price data from Alpha Vantage
            if asset_class == "Commodities":
                # For commodities, use commodity-specific endpoint
                commodity_client = CommodityData(av_client)
                price_data = commodity_client.get_commodity_data(ticker)
            elif asset_class == "Currencies":
                # For FX pairs, use forex endpoint
                # Parse currency pair (e.g., 'EUR/USD')
                if '/' in ticker:
                    from_currency, to_currency = ticker.split('/')
                else:
                    # Default to USD as quote currency
                    from_currency, to_currency = ticker, 'USD'
                
                price_data = av_client.get_fx_rate(from_currency, to_currency)
            else:
                # For other assets, use daily adjusted
                price_data = av_client.get_daily_adjusted(ticker)
            
            # Calculate returns
            if 'returns' in price_data.columns:
                returns = price_data['returns']
            else:
                # Use close price for returns calculation
                close_prices = price_data['close'] if 'close' in price_data.columns else price_data['value']
                returns = close_prices.pct_change()
            
            # Generate features based on asset class
            if asset_class == "Commodities":
                # Get the main price column
                price_col = 'close' if 'close' in price_data.columns else 'value'
                prices = price_data[price_col]
                
                # Generate commodity features
                features = calculate_commodity_features(prices, returns)
                
                # Extract specific indicators similar to AlphaVantage format
                indicators[ticker] = {
                    'SMA_20': features.filter(like='mean_20'),
                    'SMA_50': features.filter(like='mean_60'),
                    'SMA_200': features.filter(like='mean_60'),  # Not enough data for 200
                    'EMA_20': prices.ewm(span=20).mean(),
                    'RSI_14': calculate_volatility_features(returns)['rv_20d'],  # Realized volatility as substitute
                    'MACD': returns.ewm(span=12).mean() - returns.ewm(span=26).mean()
                }
                
            elif asset_class == "Currencies":
                # Get the close price for FX
                price_col = 'close' if 'close' in price_data.columns else price_data.columns[0]
                prices = price_data[price_col]
                
                # Generate FX features
                features = calculate_fx_features(prices, returns)
                
                # Extract specific indicators similar to AlphaVantage format
                indicators[ticker] = {
                    'SMA_20': features.filter(like='mean_20'),
                    'SMA_50': features.filter(like='mean_60'),
                    'SMA_200': features.filter(like='mean_60'),  # Not enough data for 200
                    'RSI_14': calculate_volatility_features(returns)['rv_20d'],  # Realized volatility as substitute
                    'MACD': returns.ewm(span=12).mean() - returns.ewm(span=26).mean()
                }
                
            elif asset_class == "Bonds":
                # Get the close price for bonds
                price_col = 'close' if 'close' in price_data.columns else price_data.columns[0]
                prices = price_data[price_col]
                
                # Generate bond features
                features = calculate_bond_features(prices, returns)
                
                # Extract specific indicators similar to AlphaVantage format
                indicators[ticker] = {
                    'SMA_20': features.filter(like='mean_20'),
                    'SMA_50': features.filter(like='mean_60'),
                    'SMA_200': features.filter(like='mean_60'),  # Not enough data for 200
                    'RSI_14': calculate_volatility_features(returns)['rv_20d'],  # Realized volatility as substitute
                    'MACD': returns.ewm(span=12).mean() - returns.ewm(span=26).mean()
                }
                
            else:  # Equities - AlphaVantage should have these, but just in case
                # Get the close price
                prices = price_data['close']
                
                # Standard indicators
                indicators[ticker] = {
                    'SMA_20': prices.rolling(window=20).mean(),
                    'SMA_50': prices.rolling(window=50).mean(),
                    'SMA_200': prices.rolling(window=200).mean(),
                    'EMA_20': prices.ewm(span=20).mean(),
                    'RSI_14': calculate_volatility_features(returns)['rv_14d'] if 'rv_14d' in calculate_volatility_features(returns) else None,
                    'MACD': prices.ewm(span=12).mean() - prices.ewm(span=26).mean()
                }
            
            logger.info(f"Generated technical indicators for {ticker}")
            
        except Exception as e:
            logger.error(f"Error generating indicators for {ticker}: {str(e)}")
            indicators[ticker] = {}
    
    return indicators

def calculate_term_structure(df, columns=None):
    """
    Calculate term structure features for commodities and bonds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price or yield data for different maturities
    columns : list, optional
        Columns representing different maturities (in ascending order)
    
    Returns:
    --------
    pd.Series
        Term structure slope
    """
    # If columns not provided, try to infer from common patterns
    if columns is None:
        # For bonds, look for typical maturity columns
        if any(col.endswith('Y') for col in df.columns):
            # Try to find common bond maturity columns (2Y, 5Y, 10Y, 30Y)
            candidate_columns = [col for col in df.columns if col.endswith('Y')]
            # Sort by maturity (assuming format like '2Y', '5Y', etc.)
            try:
                columns = sorted(candidate_columns, 
                                key=lambda x: int(x.replace('Y', '')))
            except:
                # If parsing fails, use as-is
                columns = candidate_columns
        else:
            # For other assets, use any numeric columns
            columns = df.select_dtypes(include=[np.number]).columns[:2]
            if len(columns) < 2:
                return pd.Series(0, index=df.index)  # Not enough data
    
    # Need at least two columns to calculate slope
    if len(columns) < 2:
        return pd.Series(0, index=df.index)
        
    # Calculate slope (simple difference between longest and shortest maturity)
    short_end = df[columns[0]]
    long_end = df[columns[-1]]
    
    # For bonds: lower yields at long end means positive slope
    # For commodities: higher prices at long end means positive slope (contango)
    slope = long_end - short_end
    
    # Normalize by current value to get comparable values
    normalized_slope = slope / short_end
    
    return normalized_slope

def calculate_carry(df, asset_class='Commodities'):
    """
    Calculate carry for currencies and commodities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price and other data
    asset_class : str
        Asset class ('Currencies' or 'Commodities')
    
    Returns:
    --------
    pd.Series
        Carry estimate
    """
    if asset_class == 'Currencies':
        # For currencies, carry is related to interest rate differential
        # We approximate this with forward premium/discount
        if 'forward_rate' in df.columns and 'spot_rate' in df.columns:
            # Calculate from forward rates if available
            carry = (df['forward_rate'] - df['spot_rate']) / df['spot_rate']
        else:
            # Estimate from rolling returns as a proxy
            carry = df['returns'].rolling(window=252).mean() * 252
            
    elif asset_class == 'Commodities':
        # For commodities, carry is related to convenience yield and storage costs
        # We approximate with term structure
        if 'term_structure' in df.columns:
            # Negative of term structure is a proxy for carry
            carry = -df['term_structure']
        else:
            # Use rolling returns as a proxy
            carry = df['returns'].rolling(window=126).mean() * 252
    else:
        # Default placeholder
        carry = pd.Series(0, index=df.index)
    
    return carry

def identify_market_regime(df, window=63):
    """
    Identify market volatility and trend regimes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price and returns data
    window : int
        Lookback window for regime detection
    
    Returns:
    --------
    pd.Series
        Regime indicator (-1 for high volatility/bear, 0 for neutral, 1 for low volatility/bull)
    """
    # Get returns if available, otherwise calculate from prices
    if 'returns' in df.columns:
        returns = df['returns']
    elif 'close' in df.columns:
        returns = df['close'].pct_change()
    else:
        # Use the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            returns = df[numeric_cols[0]].pct_change()
        else:
            # No usable data
            return pd.Series(0, index=df.index)
    
    # Calculate rolling volatility
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Calculate rolling returns
    ret = returns.rolling(window=window).sum()
    
    # Create regime indicator
    regime = pd.Series(0, index=df.index)
    
    # High volatility + negative returns = bear market (-1)
    regime[((vol > vol.quantile(0.8)) & (ret < 0))] = -1
    
    # Low volatility + positive returns = bull market (1)
    regime[((vol < vol.quantile(0.2)) & (ret > 0))] = 1
    
    return regime

def add_return_prediction_features(features_df, asset_class):
    """
    Add features specifically designed for return prediction (not just direction).
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature dataframe to enhance
    asset_class : str
        Asset class ('Equities', 'Bonds', 'Commodities', 'Currencies')
    
    Returns:
    --------
    pd.DataFrame
        Enhanced feature dataframe with return-predictive features
    """
    # Get price data if available (close or value)
    price_col = None
    for col_name in ['close', 'value', 'price']:
        if col_name in features_df.columns:
            price_col = col_name
            break
    
    if price_col is None:
        # Try to find any numeric column
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            # No usable data
            return features_df
    
    # Copy to avoid modifying the original
    enhanced_df = features_df.copy()
    
    # Add momentum features with proven return predictability
    for window in [60, 120, 252]:
        if len(enhanced_df) > window:  # Only if we have enough data
            enhanced_df[f'momentum_{window}d'] = enhanced_df[price_col].pct_change(periods=window)
    
    # Add term structure features for commodities and bonds
    if asset_class in ['Commodities', 'Bonds']:
        enhanced_df['term_structure'] = calculate_term_structure(enhanced_df)
        
    # Add carry features for currencies and commodities
    if asset_class in ['Currencies', 'Commodities']:
        enhanced_df['carry'] = calculate_carry(enhanced_df, asset_class)
    
    # Add liquidity and volatility regime features
    enhanced_df['regime_indicator'] = identify_market_regime(enhanced_df)
    
    # Add return reversal features
    for window in [5, 10, 20]:
        if len(enhanced_df) > window:
            enhanced_df[f'reversal_{window}d'] = -enhanced_df[price_col].pct_change(periods=window)
    
    # Add volatility scaling features - higher returns during low vol
    vol_windows = [21, 63, 126]
    for window in vol_windows:
        if len(enhanced_df) > window and 'returns' in enhanced_df.columns:
            vol = enhanced_df['returns'].rolling(window=window).std() * np.sqrt(252)
            enhanced_df[f'inverse_vol_{window}d'] = 1 / (vol + 0.001)  # Avoid division by zero
    
    return enhanced_df

# Integration with CudaOptimizedTrendPredictor
def patch_trend_predictor_class():
    """
    Patch the CudaOptimizedTrendPredictor class to use advanced features.
    This function adds the feature enhancement capabilities to the predictor class.
    
    Returns:
    --------
    None
    """
    import types
    from cuda_optimized_trend_predictor import CudaOptimizedTrendPredictor
    
    # Add feature engineering method to the class
    def enhance_with_advanced_features(self, n_features=20, normalize_method="asset_class"):
        """
        Enhance predictor with advanced features for asset classes.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to select for each asset
        normalize_method : str
            Method for normalizing features
            
        Returns:
        --------
        bool
            True if enhancement was successful
        """
        enhancer = FeatureEngineeringEnhancer(
            predictor=self,
            n_features=n_features,
            normalize_method=normalize_method
        )
        
        return enhancer.enhance_predictor()
    
    # Add generate technical indicators method to the class
    def generate_technical_indicators(self, use_cache=True):
        """
        Generate technical indicators for all assets using advanced features.
        This is a replacement for the standard technical indicators from Alpha Vantage.
        
        Parameters:
        -----------
        use_cache : bool
            Whether to use cached indicators
            
        Returns:
        --------
        dict
            Dictionary of technical indicators
        """
        # Get tickers by asset class
        tickers = []
        asset_classes = {}
        
        for asset_class, asset_tickers in self.assets_dict.items():
            tickers.extend(asset_tickers)
            for ticker in asset_tickers:
                asset_classes[ticker] = asset_class
        
        # Generate indicators using the helper function
        return enhance_technical_indicators(self.av_client, tickers, asset_classes)
    
    # Patch the class
    CudaOptimizedTrendPredictor.enhance_with_advanced_features = enhance_with_advanced_features
    CudaOptimizedTrendPredictor.generate_technical_indicators = generate_technical_indicators
    
    logger.info("Patched CudaOptimizedTrendPredictor with advanced feature capabilities")

# Always patch the class when this module is imported
patch_trend_predictor_class()