# currency_patch.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def apply_currency_patches(alpha_vantage_client_class):
    """Apply patches to the AlphaVantageClient class to handle currency symbols properly."""
    logger.info("Applying currency handling patches to AlphaVantageClient")
    
    # List of currency symbols to catch
    CURRENCY_SYMBOLS = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']
    
    # Store original methods
    original_get_daily = alpha_vantage_client_class.get_daily
    original_get_daily_adjusted = alpha_vantage_client_class.get_daily_adjusted
    original_get_technical_indicator = alpha_vantage_client_class.get_technical_indicator
    
    def patched_get_daily_adjusted(self, symbol, **kwargs):
        """Patched version that handles currency symbols."""
        if symbol in CURRENCY_SYMBOLS:
            logger.info(f"Currency symbol {symbol} detected, redirecting to FX data")
            # Determine quote currency based on convention
            if symbol in ['JPY', 'CAD', 'CHF']:
                from_currency, to_currency = 'USD', symbol
            else:
                from_currency, to_currency = symbol, 'USD'
            
            # Get FX data instead
            return self.get_fx_rate(from_currency, to_currency, interval='daily', **kwargs)
        else:
            # Call original method for non-currency symbols
            return original_get_daily_adjusted(self, symbol, **kwargs)
    
    def patched_get_daily(self, symbol, **kwargs):
        """Patched version that handles currency symbols."""
        if symbol in CURRENCY_SYMBOLS:
            logger.info(f"Currency symbol {symbol} detected, redirecting to FX data")
            # Determine quote currency based on convention
            if symbol in ['JPY', 'CAD', 'CHF']:
                from_currency, to_currency = 'USD', symbol
            else:
                from_currency, to_currency = symbol, 'USD'
            
            # Get FX data instead
            return self.get_fx_rate(from_currency, to_currency, interval='daily', **kwargs)
        else:
            # Call original method for non-currency symbols
            return original_get_daily(self, symbol, **kwargs)
    
    def patched_get_technical_indicator(self, symbol, indicator, **kwargs):
        """Patched version that handles currency symbols."""
        if symbol in CURRENCY_SYMBOLS:
            logger.info(f"Currency symbol {symbol} detected, calculating {indicator} locally")
            try:
                # Determine quote currency based on convention
                if symbol in ['JPY', 'CAD', 'CHF']:
                    from_currency, to_currency = 'USD', symbol
                else:
                    from_currency, to_currency = symbol, 'USD'
                
                # Get FX data
                fx_data = self.get_fx_rate(from_currency, to_currency, interval='daily', cache=True)
                
                # Calculate indicator locally
                if indicator == 'SMA':
                    period = kwargs.get('time_period', 20)
                    return pd.DataFrame({
                        'SMA': fx_data['close'].rolling(window=period).mean()
                    })
                elif indicator == 'EMA':
                    period = kwargs.get('time_period', 20)
                    return pd.DataFrame({
                        'EMA': fx_data['close'].ewm(span=period, adjust=False).mean()
                    })
                elif indicator == 'RSI':
                    period = kwargs.get('time_period', 14)
                    # Calculate RSI
                    delta = fx_data['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
                    rsi = 100 - (100 / (1 + rs))
                    return pd.DataFrame({'RSI': rsi})
                elif indicator == 'MACD':
                    # Use standard parameters
                    fast = 12
                    slow = 26
                    signal = 9
                    # Calculate MACD
                    fast_ema = fx_data['close'].ewm(span=fast, adjust=False).mean()
                    slow_ema = fx_data['close'].ewm(span=slow, adjust=False).mean()
                    macd_line = fast_ema - slow_ema
                    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
                    histogram = macd_line - signal_line
                    return pd.DataFrame({
                        'MACD': macd_line,
                        'MACD_Signal': signal_line,
                        'MACD_Hist': histogram
                    })
                elif indicator == 'BBANDS':
                    period = kwargs.get('time_period', 20)
                    # Calculate Bollinger Bands
                    middle_band = fx_data['close'].rolling(window=period).mean()
                    std_dev = fx_data['close'].rolling(window=period).std()
                    upper_band = middle_band + (std_dev * 2)
                    lower_band = middle_band - (std_dev * 2)
                    return pd.DataFrame({
                        'Middle Band': middle_band,
                        'Upper Band': upper_band,
                        'Lower Band': lower_band
                    })
                else:
                    # For other indicators, try original method (will likely fail)
                    logger.warning(f"Unsupported indicator {indicator} for currency {symbol}")
                    return original_get_technical_indicator(self, symbol, indicator, **kwargs)
            except Exception as e:
                logger.error(f"Error calculating {indicator} for currency {symbol}: {str(e)}")
                # Return empty DataFrame with appropriate structure
                if indicator == 'MACD':
                    return pd.DataFrame(columns=['MACD', 'MACD_Signal', 'MACD_Hist'])
                elif indicator == 'BBANDS':
                    return pd.DataFrame(columns=['Middle Band', 'Upper Band', 'Lower Band'])
                else:
                    return pd.DataFrame(columns=[indicator])
        else:
            # Call original method for non-currency symbols
            return original_get_technical_indicator(self, symbol, indicator, **kwargs)
    
    # Apply the patches
    alpha_vantage_client_class.get_daily_adjusted = patched_get_daily_adjusted
    alpha_vantage_client_class.get_daily = patched_get_daily
    alpha_vantage_client_class.get_technical_indicator = patched_get_technical_indicator
    
    logger.info("Currency handling patches applied successfully")