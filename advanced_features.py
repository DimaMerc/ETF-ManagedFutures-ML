# advanced_features.py
"""
Advanced feature engineering for algorithmic trading
Implements specialized features for commodities, FX pairs, and bonds
based on research recommendations for improving predictive power.

These features can be calculated directly from price/returns data
without requiring technical indicators from Alpha Vantage.
"""

import numpy as np
import pandas as pd
import pywt  # PyWavelets for wavelet transform
from scipy import stats, signal
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Make VMD features optional with fallbacks
HAS_VMD = False
try:
    # Check if vmdpy is available
    import vmdpy
    HAS_VMD = True
except ImportError:
    HAS_VMD = False
    print("VMD package not found. Using wavelet transform as alternative.")


###########################################
# Common Statistical Features (All Assets)
###########################################

def calculate_statistical_moments(series, windows=[5, 10, 20, 60]):
    """
    Calculate statistical moments (mean, variance, skewness, kurtosis) over different time windows.
    
    Parameters:
    -----------
    series : pd.Series
        Price or returns series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with statistical moments
    """
    results = pd.DataFrame(index=series.index)
    
    # Create features for each window
    for window in windows:
        # Mean (first moment)
        results[f'mean_{window}'] = series.rolling(window=window).mean()
        
        # Variance (second moment)
        results[f'variance_{window}'] = series.rolling(window=window).var()
        
        # Skewness (third moment) - asymmetry
        results[f'skew_{window}'] = series.rolling(window=window).skew()
        
        # Kurtosis (fourth moment) - tail extremity
        results[f'kurt_{window}'] = series.rolling(window=window).kurt()
        
        # Z-score (normalized deviations)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        results[f'zscore_{window}'] = (series - rolling_mean) / rolling_std
    
    return results

def calculate_returns_features(prices, windows=[1, 5, 10, 20, 60]):
    """
    Calculate various return-based features: log returns, momentum, and acceleration.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with return-based features
    """
    results = pd.DataFrame(index=prices.index)
    
    # Log returns
    results['log_return_1d'] = np.log(prices / prices.shift(1))
    
    # Calculate returns, momentum, and acceleration for different windows
    for window in windows:
        # Simple returns
        results[f'return_{window}d'] = prices.pct_change(periods=window)
        
        # Log returns over window
        results[f'log_return_{window}d'] = np.log(prices / prices.shift(window))
        
        # Momentum (velocity) - rate of change
        results[f'momentum_{window}d'] = (prices - prices.shift(window)) / window
        
        # Acceleration - change in momentum
        if window > 1:
            momentum = (prices - prices.shift(window)) / window
            results[f'acceleration_{window}d'] = momentum - momentum.shift(window)
    
    return results

# In advanced_features.py, replace the calculate_volatility_features function:

def calculate_volatility_features(returns, windows=[5, 10, 20, 60]):
    """
    Calculate volatility features with improved error handling.
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volatility features
    """
    # Ensure returns is a simple Series of numbers
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            # Use first numeric column
            numeric_cols = returns.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                returns = returns[numeric_cols[0]]
            else:
                # Default to first column
                returns = returns.iloc[:, 0]
    
    # Convert to numeric, coerce errors to NaN
    returns = pd.to_numeric(returns, errors='coerce')
    
    results = pd.DataFrame(index=returns.index)
    
    # Create volatility features for each window
    for window in windows:
        try:
            # Standard deviation
            results[f'std_{window}d'] = returns.rolling(window=window).std()
            
            # Realized volatility (RV) - SIMPLIFIED VERSION THAT AVOIDS APPLY()
            squared_returns = returns ** 2
            results[f'rv_{window}d'] = np.sqrt(squared_returns.rolling(window=window).sum())
            
            # Bi-power variation (BPV) - robust to jumps
            abs_returns = np.abs(returns)
            bpv = abs_returns * abs_returns.shift(1)
            results[f'bpv_{window}d'] = np.sqrt((np.pi/2) * bpv.rolling(window=window).sum())
            
            # Jump component (difference between RV and BPV)
            if f'rv_{window}d' in results.columns and f'bpv_{window}d' in results.columns:
                results[f'jump_{window}d'] = np.maximum(
                    0, results[f'rv_{window}d'] - results[f'bpv_{window}d'])
        except Exception as e:
            # In case of error, fill with NaN
            results[f'std_{window}d'] = np.nan
            results[f'rv_{window}d'] = np.nan
            results[f'bpv_{window}d'] = np.nan
            results[f'jump_{window}d'] = np.nan
    
    return results

def detect_price_jumps(series, window=20, threshold=3.0):
    """
    Detect price jumps that exceed a threshold based on local volatility.
    
    Parameters:
    -----------
    series : pd.Series
        Price or returns series
    window : int
        Window size for local volatility calculation
    threshold : float
        Number of standard deviations to identify jumps
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with jump indicators and sizes
    """
    results = pd.DataFrame(index=series.index)
    
    # Calculate returns if Series is price data
    if series.min() > 0 and series.pct_change().mean() < 0.1:  # Likely price data
        returns = series.pct_change()
    else:  # Already returns data
        returns = series
    
    # Calculate local volatility
    rolling_std = returns.rolling(window=window).std()
    
    # Identify jumps
    jumps = (returns.abs() > (threshold * rolling_std)) & (rolling_std > 0)
    
    # Store jump indicators and sizes
    results['jump_indicator'] = jumps.astype(int)
    results['jump_size'] = returns * jumps
    
    # Jump metrics over multiple windows
    for w in [5, 10, 20]:
        results[f'jump_count_{w}d'] = jumps.rolling(window=w).sum()
        results[f'jump_intensity_{w}d'] = results['jump_size'].abs().rolling(window=w).sum()
    
    return results

def calculate_time_series_features(series, windows=[10, 20, 60]):
    """
    Calculate time series characteristics: autocorrelation, GARCH effects,
    and mean-reverting vs. trending behavior.
    
    Parameters:
    -----------
    series : pd.Series
        Price or returns series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with time series features
    """
    results = pd.DataFrame(index=series.index)
    
    # Create time series features for each window
    for window in windows:
        # Run autocorrelation for each window of data
        results[f'autocorr_lag1_{window}d'] = series.rolling(window=window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        
        # ARCH effect (autocorrelation of squared returns)
        squared = series ** 2
        results[f'arch_effect_{window}d'] = squared.rolling(window=window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        
        # Hurst exponent (simplified calculation)
        # H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
        results[f'hurst_{window}d'] = series.rolling(window=window).apply(
            lambda x: calculate_hurst_exponent(x), raw=False)
        
        # Trending strength - ratio of current value to moving average
        ma = series.rolling(window=window).mean()
        results[f'trend_strength_{window}d'] = series / ma
        
    return results

def calculate_hurst_exponent(series, lags=None):
    """
    Calculate Hurst exponent - indicator of mean-reverting vs trending behavior.
    H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
    
    Parameters:
    -----------
    series : array-like
        Time series data
    lags : list, optional
        List of lag values
        
    Returns:
    --------
    float
        Hurst exponent
    """
    # Convert to numpy array if it's not
    series = np.array(series)
    
    # Remove NaN values
    series = series[~np.isnan(series)]
    
    # Need at least 10 points for meaningful calculation
    if len(series) < 10:
        return np.nan
    
    # Set default lags if not provided
    if lags is None:
        lags = range(2, min(20, len(series) // 4))
    
    # Calculate tau
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    
    # Avoid log(0) or division by zero
    if any(t <= 0 for t in tau) or len(tau) < 2:
        return np.nan
    
    # Fit line to log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return Hurst exponent
    return m[0]


###########################################
# Commodity-Specific Features
###########################################

def calculate_commodity_features(prices, returns=None):
    """
    Calculate specialized features for commodity price prediction.
    
    Parameters:
    -----------
    prices : pd.Series
        Commodity price series
    returns : pd.Series, optional
        Returns series (calculated if not provided)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with commodity-specific features
    """
    # Calculate returns if not provided
    if returns is None:
        returns = prices.pct_change()
    
    # Start with common features
    results = pd.DataFrame(index=prices.index)
    
    # Add statistical moments
    stat_moments = calculate_statistical_moments(prices)
    results = pd.concat([results, stat_moments], axis=1)
    
    # Add returns features
    ret_features = calculate_returns_features(prices)
    results = pd.concat([results, ret_features], axis=1)
    
    # Add volatility and jump features
    vol_features = calculate_volatility_features(returns)
    results = pd.concat([results, vol_features], axis=1)
    
    jump_features = detect_price_jumps(returns)
    results = pd.concat([results, jump_features], axis=1)
    
    # Add time series features
    ts_features = calculate_time_series_features(returns)
    results = pd.concat([results, ts_features], axis=1)
    
    # Add seasonal decomposition
    try:
        # Detect seasonality
        if len(prices) > 24:  # Need enough data for seasonality
            seasonal_features = extract_seasonal_components(prices)
            results = pd.concat([results, seasonal_features], axis=1)
    except Exception as e:
        print(f"Error in seasonal decomposition: {str(e)}")
    
    # Add VMD features if available
    if HAS_VMD and len(prices) > 50:
        try:
            vmd_features = calculate_vmd_features(prices.values)
            # Convert to DataFrame with same index
            vmd_df = pd.DataFrame(vmd_features, index=prices.index[:len(vmd_features[next(iter(vmd_features))])])
            results = pd.concat([results, vmd_df], axis=1)
        except Exception as e:
            print(f"Error in VMD decomposition: {str(e)}")
    
    # Add wavelet features
    try:
        if len(prices) >= 64:  # Need at least 64 points for wavelets
            wavelet_features = calculate_wavelet_features(prices)
            results = pd.concat([results, wavelet_features], axis=1)
    except Exception as e:
        print(f"Error in wavelet decomposition: {str(e)}")
    
    return results

def extract_seasonal_components(series, period=None):
    """
    Extract seasonal, trend, and residual components using STL decomposition.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    period : int, optional
        Seasonality period (auto-detected if None)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with seasonal components
    """
    # Fill missing values for decomposition
    series_filled = series.fillna(method='ffill').fillna(method='bfill')
    
    # Detect period if not provided (assumes daily data)
    if period is None:
        # Try to infer frequency from the data
        if hasattr(series, 'index') and hasattr(series.index, 'freq'):
            freq = series.index.freq
            if freq == 'D' or freq == 'B':  # Daily or business day
                period = 252  # Trading days in a year
            elif freq == 'M':  # Monthly
                period = 12
            elif freq == 'W':  # Weekly
                period = 52
            else:
                period = 252  # Default for financial data
        else:
            # Default period heuristic - choose between weekly, monthly, quarterly
            if len(series) < 120:
                period = 7  # Weekly
            elif len(series) < 500:
                period = 30  # Monthly
            else:
                period = 252  # Yearly for financial data
    
    # Adjust period to be reasonable given data length
    if period > len(series) // 4:
        period = len(series) // 4
    
    if period < 2:
        period = 2
    
    # Perform STL decomposition
    try:
        stl = STL(series_filled, period=period, robust=True)
        result = stl.fit()
        
        # Create features
        seasonal = result.seasonal
        trend = result.trend
        residual = result.resid
        
        # Calculate seasonal strength
        seasonal_strength = 1 - (np.var(residual) / np.var(seasonal + residual))
        seasonal_strength = max(0, seasonal_strength)  # Ensure non-negative
        
        # Calculate trend strength
        trend_strength = 1 - (np.var(residual) / np.var(trend + residual))
        trend_strength = max(0, trend_strength)  # Ensure non-negative
        
        # Create features DataFrame
        features = pd.DataFrame({
            'seasonal': seasonal,
            'trend': trend,
            'residual': residual,
            'seasonal_strength': np.repeat(seasonal_strength, len(series)),
            'trend_strength': np.repeat(trend_strength, len(series)),
            'detrended': series_filled - trend
        }, index=series.index)
        
        return features
        
    except Exception as e:
        print(f"STL decomposition failed: {str(e)}")
        # Return empty DataFrame with same index
        return pd.DataFrame(index=series.index)

def calculate_vmd_features(data, K=3):
    """
    Decompose time series using Variational Mode Decomposition (VMD).
    Requires vmdpy package to be installed.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    K : int
        Number of modes to extract
        
    Returns:
    --------
    dict
        Dictionary with VMD components
    """
    if not HAS_VMD:
        # Fallback to simple decomposition if VMD not available
        return simple_decomposition(data, n_components=K)
    
    # Ensure data is a numpy array
    data = np.array(data).flatten()
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    # Default VMD parameters
    alpha = 2000      # Moderate bandwidth constraint
    tau = 0           # Noise-tolerance (no noise)
    DC = 0            # No DC part imposed
    init = 1          # Initialize omegas uniformly
    tol = 1e-7        # Tolerance
    
    # Perform VMD decomposition using vmdpy if available
    try:
        from vmdpy import VMD
        _, u_hat, _ = VMD(data, alpha, tau, K, DC, init, tol)
        
        # Extract modes
        modes = np.real(np.fft.ifft(u_hat))
        
        # Create features dictionary
        features = {}
        for i in range(K):
            features[f'vmd_mode_{i+1}'] = modes[i, :]
        
        # Calculate mode energies (variance contribution)
        for i in range(K):
            features[f'vmd_energy_{i+1}'] = np.var(modes[i, :]) / np.var(data)
        
        # Calculate combined features
        features['vmd_trend'] = modes[0, :]  # Lowest frequency component is trend
        features['vmd_cycle'] = np.sum(modes[1:, :], axis=0)  # Combine higher modes
        
        return features
    except:
        # Fallback to simple decomposition if VMD fails
        return simple_decomposition(data, n_components=K)

def simple_decomposition(data, n_components=3):
    """
    Simple time series decomposition as a fallback when VMD is not available.
    Uses moving averages of different lengths.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    n_components : int
        Number of components to extract
        
    Returns:
    --------
    dict
        Dictionary with decomposition components
    """
    # Ensure data is a numpy array
    data = np.array(data).flatten()
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    # Calculate moving averages of different lengths
    ma_lengths = [int(len(data) * (i+1) / (n_components+2)) for i in range(n_components)]
    ma_lengths = [max(3, length) for length in ma_lengths]  # Ensure minimum length
    
    # Create features dictionary
    features = {}
    
    # Calculate moving averages
    for i, length in enumerate(ma_lengths):
        # Create padded data to maintain same length as input
        padded = np.pad(data, (length//2, length//2), mode='edge')
        
        # Apply moving average
        ma = np.convolve(padded, np.ones(length)/length, mode='valid')
        
        # Trim to original length if needed
        if len(ma) > len(data):
            ma = ma[:len(data)]
        elif len(ma) < len(data):
            ma = np.pad(ma, (0, len(data) - len(ma)), mode='edge')
        
        # Store component
        features[f'vmd_mode_{i+1}'] = ma
        
        # Calculate energy
        features[f'vmd_energy_{i+1}'] = np.var(ma) / np.var(data)
    
    # Calculate residual
    residual = data - features['vmd_mode_1']
    
    # Create trend and cycle
    features['vmd_trend'] = features['vmd_mode_1']  # Longest MA is trend
    
    # Create cycle as combination of higher frequency components
    if n_components > 1:
        features['vmd_cycle'] = residual
    else:
        features['vmd_cycle'] = np.zeros_like(data)
    
    return features

def calculate_wavelet_features(series, wavelet='db4', level=3):
    """
    Calculate wavelet transform features.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    wavelet : str
        Wavelet type (default: 'db4')
    level : int
        Decomposition level
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with wavelet features
    """
    # Ensure data is a numpy array
    data = series.values
    
    # Pad data to power of 2 for efficiency
    original_length = len(data)
    power = int(np.ceil(np.log2(original_length)))
    padded_length = 2**power
    padded_data = np.pad(data, (0, padded_length - original_length), mode='constant')
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(padded_data, wavelet, level=level)
    
    # Extract approximation and detail coefficients
    a = coeffs[0]  # Approximation coefficients
    d = coeffs[1:]  # Detail coefficients
    
    # Create features DataFrame
    features = pd.DataFrame(index=series.index)
    
    # Reconstruct each level to original length
    features['wavelet_approx'] = pywt.upcoef('a', a, wavelet, level=level)[:original_length]
    
    for i, detail in enumerate(d):
        # Reconstruct detail coefficients for each level
        detail_recon = pywt.upcoef('d', detail, wavelet, level=level-i)[:original_length]
        features[f'wavelet_detail_{i+1}'] = detail_recon
        
        # Calculate energy of detail coefficients
        features[f'wavelet_energy_{i+1}'] = np.sum(detail**2) / np.sum(padded_data**2)
    
    # Calculate wavelet entropy
    total_energy = sum(np.sum(c**2) for c in coeffs)
    if total_energy > 0:
        relative_energies = [np.sum(c**2) / total_energy for c in coeffs]
        entropy = -sum(e * np.log2(e) if e > 0 else 0 for e in relative_energies)
        features['wavelet_entropy'] = np.repeat(entropy, len(series))
    
    return features


###########################################
# FX Pair Features
###########################################

def calculate_fx_features(prices, returns=None):
    """
    Calculate specialized features for FX pair prediction.
    
    Parameters:
    -----------
    prices : pd.Series
        FX rate series
    returns : pd.Series, optional
        Returns series (calculated if not provided)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with FX-specific features
    """
    # Calculate returns if not provided
    if returns is None:
        returns = prices.pct_change()
    
    # Start with common features
    results = pd.DataFrame(index=prices.index)
    
    # Add price transformation features
    features = calculate_returns_features(prices)
    results = pd.concat([results, features], axis=1)
    
    # Add time series characteristics
    ts_features = calculate_time_series_features(returns, windows=[5, 10, 20, 60])
    results = pd.concat([results, ts_features], axis=1)
    
    # Add autocorrelation features with more lags
    acf_features = calculate_autocorrelation_features(returns)
    results = pd.concat([results, acf_features], axis=1)
    
    # Add Hurst exponent for mean-reversion/trending
    hurst_features = calculate_hurst_features(returns)
    results = pd.concat([results, hurst_features], axis=1)
    
    # Add volatility and jump features
    vol_features = calculate_volatility_features(returns)
    results = pd.concat([results, vol_features], axis=1)
    
    jump_features = detect_price_jumps(returns)
    results = pd.concat([results, jump_features], axis=1)
    
    # Add time-of-day cyclical encoding if datetime index is available
    if hasattr(prices, 'index') and isinstance(prices.index, pd.DatetimeIndex):
        try:
            time_features = calculate_time_cyclical_features(prices)
            results = pd.concat([results, time_features], axis=1)
        except Exception as e:
            print(f"Error calculating time features: {str(e)}")
    
    return results

def calculate_autocorrelation_features(returns, max_lag=10):
    """
    Calculate autocorrelation features at various lags.
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
    max_lag : int
        Maximum lag to calculate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with autocorrelation features
    """
    results = pd.DataFrame(index=returns.index)
    
    # Calculate autocorrelation for each window
    for window in [30, 60, 90]:
        for lag in range(1, min(max_lag+1, window//2)):
            # Calculate rolling autocorrelation
            results[f'acf_lag{lag}_{window}d'] = returns.rolling(window=window).apply(
                lambda x: acf(x, nlags=lag)[-1] if len(x.dropna()) > lag else np.nan, raw=False)
            
            # Calculate rolling partial autocorrelation
            try:
                results[f'pacf_lag{lag}_{window}d'] = returns.rolling(window=window).apply(
                    lambda x: pacf(x, nlags=lag)[-1] if len(x.dropna()) > lag else np.nan, raw=False)
            except:
                # Simplified calculation if pacf fails
                results[f'pacf_lag{lag}_{window}d'] = returns.rolling(window=window).apply(
                    lambda x: pd.Series(x).autocorr(lag=lag) if len(x.dropna()) > lag else np.nan, raw=False)
    
    # Calculate sign autocorrelation (direction persistence)
    signed_returns = np.sign(returns)
    for lag in range(1, max_lag+1):
        results[f'sign_autocorr_lag{lag}'] = signed_returns * signed_returns.shift(lag)
    
    return results

def calculate_hurst_features(returns, windows=[30, 60, 90, 120]):
    """
    Calculate Hurst exponent over different windows to identify trends/mean-reversion.
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Hurst exponent features
    """
    results = pd.DataFrame(index=returns.index)
    
    # Calculate Hurst exponent for different windows
    for window in windows:
        # Skip if window too small
        if window < 20:
            continue
            
        results[f'hurst_{window}d'] = returns.rolling(window=window).apply(
            lambda x: calculate_hurst_exponent(x), raw=False)
        
        # Add trend/mean-reversion indicator (1 for trend, -1 for mean-reversion)
        hurst = results[f'hurst_{window}d']
        results[f'trend_indicator_{window}d'] = np.where(hurst > 0.55, 1,
                                               np.where(hurst < 0.45, -1, 0))
        
        # Calculate trend strength
        results[f'trend_strength_{window}d'] = np.abs(hurst - 0.5) * 2  # Scaled 0-1
    
    return results

def calculate_time_cyclical_features(series):
    """
    Calculate cyclical time features (hour of day, day of week, etc.).
    
    Parameters:
    -----------
    series : pd.Series
        Time series with DatetimeIndex
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cyclical time features
    """
    results = pd.DataFrame(index=series.index)
    
    # Extract datetime components
    if hasattr(series, 'index') and isinstance(series.index, pd.DatetimeIndex):
        idx = series.index
        
        # Hour of day (0-23) - cyclical encoding
        if hasattr(idx, 'hour'):
            hour = idx.hour
            results['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            results['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6) - cyclical encoding
        if hasattr(idx, 'dayofweek'):
            dow = idx.dayofweek
            results['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            results['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Day of month (1-31) - cyclical encoding
        if hasattr(idx, 'day'):
            day = idx.day
            results['dom_sin'] = np.sin(2 * np.pi * day / 31)
            results['dom_cos'] = np.cos(2 * np.pi * day / 31)
        
        # Month of year (1-12) - cyclical encoding
        if hasattr(idx, 'month'):
            month = idx.month
            results['month_sin'] = np.sin(2 * np.pi * month / 12)
            results['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    return results


###########################################
# Bond/Fixed Income Features
###########################################

def calculate_bond_features(prices, returns=None, yield_data=None):
    """
    Calculate specialized features for bond ETF prediction.
    
    Parameters:
    -----------
    prices : pd.Series
        Bond ETF price series
    returns : pd.Series, optional
        Returns series (calculated if not provided)
    yield_data : pd.DataFrame, optional
        Yield curve data (if available)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with bond-specific features
    """
    # Calculate returns if not provided
    if returns is None:
        returns = prices.pct_change()
    
    # Start with common features
    results = pd.DataFrame(index=prices.index)
    
    # Add basic statistical features
    stat_features = calculate_statistical_moments(prices)
    results = pd.concat([results, stat_features], axis=1)
    
    # Add returns and momentum features
    ret_features = calculate_returns_features(prices)
    results = pd.concat([results, ret_features], axis=1)
    
    # Add time series features
    ts_features = calculate_time_series_features(returns)
    results = pd.concat([results, ts_features], axis=1)
    
    # Add volatility and jump features
    vol_features = calculate_volatility_features(returns)
    results = pd.concat([results, vol_features], axis=1)
    
    # Add yield curve features if yield data is provided
    if yield_data is not None:
        try:
            yield_features = calculate_yield_curve_features(yield_data, prices.index)
            # Align indices
            yield_features = yield_features.reindex(prices.index, method='ffill')
            results = pd.concat([results, yield_features], axis=1)
        except Exception as e:
            print(f"Error calculating yield curve features: {str(e)}")
    
    # Add bond-specific trend reversal features
    rsi_features = calculate_rsi_features(prices)
    results = pd.concat([results, rsi_features], axis=1)
    
    # Calculate bond momentum features
    bond_momentum = calculate_bond_momentum_features(prices)
    results = pd.concat([results, bond_momentum], axis=1)
    
    return results

def calculate_yield_curve_features(yield_data, target_index=None):
    """
    Calculate yield curve features: PCA components, term spreads, and butterfly spreads.
    
    Parameters:
    -----------
    yield_data : pd.DataFrame
        DataFrame with yield data for different maturities
    target_index : pd.DatetimeIndex, optional
        Target index for resampling features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with yield curve features
    """
    # Ensure yield_data has expected format
    if yield_data.empty:
        return pd.DataFrame(index=target_index if target_index is not None else pd.DatetimeIndex([]))
    
    # Use yield_data index if target_index not provided
    if target_index is None:
        target_index = yield_data.index
    
    results = pd.DataFrame(index=yield_data.index)
    
    # Calculate term spreads (difference between long and short rates)
    maturities = sorted(yield_data.columns)
    
    # Calculate all possible term spreads
    for i, short_term in enumerate(maturities[:-1]):
        for long_term in maturities[i+1:]:
            spread_name = f'term_spread_{short_term}_{long_term}'
            results[spread_name] = yield_data[long_term] - yield_data[short_term]
    
    # Calculate butterfly spreads for standard combinations
    standard_butterflies = [
        ('2Y', '5Y', '10Y'),
        ('5Y', '10Y', '30Y'),
        ('1Y', '5Y', '10Y')
    ]
    
    for short, mid, long in standard_butterflies:
        if short in maturities and mid in maturities and long in maturities:
            butterfly_name = f'butterfly_{short}_{mid}_{long}'
            results[butterfly_name] = (2 * yield_data[mid]) - (yield_data[short] + yield_data[long])
    
    # Calculate PCA components if enough data points
    if len(yield_data) > 10 and len(yield_data.columns) > 2:
        try:
            # Standardize data for PCA
            scaler = StandardScaler()
            scaled_yields = scaler.fit_transform(yield_data)
            
            # Perform PCA
            pca = PCA(n_components=min(3, len(yield_data.columns)))
            pca_result = pca.fit_transform(scaled_yields)
            
            # Add PCA components
            for i in range(pca_result.shape[1]):
                results[f'yield_pca_{i+1}'] = pca_result[:, i]
            
            # Add explained variance
            for i in range(len(pca.explained_variance_ratio_)):
                results[f'yield_pca_var_{i+1}'] = np.repeat(
                    pca.explained_variance_ratio_[i], len(yield_data))
                
            # Calculate yield curve factors with interpretation
            results['yield_curve_level'] = pca_result[:, 0]  # First PC typically represents level
            if pca_result.shape[1] > 1:
                results['yield_curve_slope'] = pca_result[:, 1]  # Second PC typically represents slope
            if pca_result.shape[1] > 2:
                results['yield_curve_curvature'] = pca_result[:, 2]  # Third PC typically represents curvature
            
        except Exception as e:
            print(f"PCA calculation failed: {str(e)}")
    
    # Add moving averages and crossovers of key spreads
    if '10Y' in maturities and '2Y' in maturities:
        key_spread = '10Y_2Y_spread'
        results[key_spread] = yield_data['10Y'] - yield_data['2Y']
        
        for window in [5, 10, 20]:
            results[f'{key_spread}_ma{window}'] = results[key_spread].rolling(window=window).mean()
        
        # Add spread reversal indicators
        results[f'{key_spread}_direction'] = np.sign(results[key_spread].diff())
    
    # Resample to target index if needed
    if target_index is not None and not target_index.equals(results.index):
        results = results.reindex(target_index, method='ffill')
    
    return results

def calculate_rsi_features(prices, windows=[5, 10, 14, 21]):
    """
    Calculate Relative Strength Index (RSI) features for different windows.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with RSI features
    """
    results = pd.DataFrame(index=prices.index)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Calculate RSI for different windows
    for window in windows:
        # Get gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add to results
        results[f'rsi_{window}'] = rsi
        
        # Add RSI-based indicators
        results[f'rsi_{window}_oversold'] = (rsi < 30).astype(int)
        results[f'rsi_{window}_overbought'] = (rsi > 70).astype(int)
        
        # RSI trend
        results[f'rsi_{window}_trend'] = np.sign(rsi.diff(5))
    
    return results

def calculate_bond_momentum_features(prices, windows=[5, 10, 20, 60]):
    """
    Calculate momentum-based features specifically for bonds.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with bond momentum features
    """
    results = pd.DataFrame(index=prices.index)
    
    # Calculate moving averages
    for window in windows:
        results[f'ma_{window}'] = prices.rolling(window=window).mean()
    
    # Calculate moving average crossovers
    for fast in windows[:-1]:
        for slow in windows[1:]:
            if fast < slow:
                fast_ma = results[f'ma_{fast}']
                slow_ma = results[f'ma_{slow}']
                
                # Crossover indicator
                results[f'ma_cross_{fast}_{slow}'] = ((fast_ma > slow_ma) & 
                                                    (fast_ma.shift(1) <= slow_ma.shift(1))).astype(int)
                results[f'ma_cross_{slow}_{fast}'] = ((fast_ma < slow_ma) & 
                                                    (fast_ma.shift(1) >= slow_ma.shift(1))).astype(int)
                
                # Moving average spread
                results[f'ma_spread_{fast}_{slow}'] = (fast_ma - slow_ma) / slow_ma
    
    # Calculate price position relative to moving averages
    for window in windows:
        ma = results[f'ma_{window}']
        results[f'price_vs_ma_{window}'] = (prices - ma) / ma
        results[f'above_ma_{window}'] = (prices > ma).astype(int)
    
    # Calculate Rate-of-Change (ROC) momentum
    for window in windows:
        results[f'roc_{window}'] = (prices / prices.shift(window) - 1) * 100
    
    return results


###########################################
# Normalization and Scaling Functions
###########################################

def normalize_features_for_lstm(features_df, method='robust', window=None):
    """
    Normalize features for LSTM model input with various methods.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features
    method : str
        Normalization method ('standard', 'minmax', 'robust', 'quantile', 'rolling')
    window : int, optional
        Window size for rolling normalization
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized features
    """
    normalized_df = pd.DataFrame(index=features_df.index)
    
    # Handle rolling normalization separately
    if method == 'rolling' and window is not None:
        for col in features_df.columns:
            # Skip columns that are already normalized or should be excluded
            if col.startswith('scaled_') or col.startswith('normalized_'):
                continue
                
            try:
                rolling_mean = features_df[col].rolling(window=window).mean()
                rolling_std = features_df[col].rolling(window=window).std()
                
                # Avoid division by zero
                normalized_df[col] = (features_df[col] - rolling_mean) / rolling_std.replace(0, 1)
                
                # Cap extreme values
                normalized_df[col] = normalized_df[col].clip(-5, 5)
            except Exception as e:
                print(f"Error normalizing column {col}: {str(e)}")
                normalized_df[col] = features_df[col]
                
        return normalized_df
    
    # Select scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif method == 'robust':
        scaler = RobustScaler(quantile_range=(5, 95))
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
    else:
        # Default to robust scaling
        scaler = RobustScaler(quantile_range=(5, 95))
    
    # Apply scaling to each column
    for col in features_df.columns:
        # Skip columns that are already normalized or should be excluded
        if col.startswith('scaled_') or col.startswith('normalized_'):
            continue
            
        try:
            # Reshape for scikit-learn API
            values = features_df[col].values.reshape(-1, 1)
            
            # Fit and transform
            scaled_values = scaler.fit_transform(values).flatten()
            
            # Store in result DataFrame
            normalized_df[col] = scaled_values
        except Exception as e:
            print(f"Error normalizing column {col}: {str(e)}")
            normalized_df[col] = features_df[col]
    
    return normalized_df

def normalize_by_volatility(features_df, returns, window=20):
    """
    Normalize features by dividing by asset volatility to prevent high-volatility
    assets from dominating the model.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for multiple assets
    returns : pd.DataFrame
        DataFrame with returns for multiple assets
    window : int
        Window size for volatility calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volatility-adjusted features
    """
    # Calculate rolling volatility for each asset
    if isinstance(returns, pd.DataFrame):
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    else:
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Create result DataFrame
    normalized_df = pd.DataFrame(index=features_df.index)
    
    # Apply volatility normalization to each column
    for col in features_df.columns:
        # Get the asset for this column
        if isinstance(features_df.columns, pd.MultiIndex):
            asset = col[0]  # For MultiIndex columns (asset, feature)
            normalized_df[col] = features_df[col] / rolling_vol[asset]
        else:
            # Assume single asset
            normalized_df[col] = features_df[col] / rolling_vol
    
    return normalized_df

def normalize_by_asset_class(features_df, asset_class_map):
    """
    Apply different normalization techniques to different asset classes.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for multiple assets
    asset_class_map : dict
        Dictionary mapping assets to asset classes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with asset-class-specific normalization
    """
    # Group assets by asset class
    assets_by_class = {}
    for asset, asset_class in asset_class_map.items():
        if asset_class not in assets_by_class:
            assets_by_class[asset_class] = []
        assets_by_class[asset_class].append(asset)
    
    # Define normalization methods for each asset class
    normalization_methods = {
        'Commodities': 'robust',     # Robust scaling for fat tails
        'Currencies': 'standard',    # Standard scaling for FX
        'Bonds': 'minmax',           # MinMax scaling for bonds (less extreme)
        'Equities': 'quantile'       # Quantile transform for equities (normalize distribution)
    }
    
    # Create result DataFrame
    normalized_df = pd.DataFrame(index=features_df.index)
    
    # Apply appropriate normalization to each asset class
    for asset_class, assets in assets_by_class.items():
        # Select method
        method = normalization_methods.get(asset_class, 'robust')
        
        # Select features for this asset class
        if isinstance(features_df.columns, pd.MultiIndex):
            # For MultiIndex columns
            class_features = features_df.loc[:, [col for col in features_df.columns if col[0] in assets]]
        else:
            # For single asset
            class_features = features_df[[col for col in features_df.columns if col in assets]]
        
        # Normalize
        normalized_class = normalize_features_for_lstm(class_features, method=method)
        
        # Add to result
        for col in normalized_class.columns:
            normalized_df[col] = normalized_class[col]
    
    return normalized_df


###########################################
# Feature Selection and Extraction
###########################################

def select_top_features(features_df, returns, n_features=20, method='correlation'):
    """
    Select top features based on correlation with returns or feature importance.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features
    returns : pd.Series
        Returns series
    n_features : int
        Number of features to select
    method : str
        Selection method ('correlation', 'mutual_info', 'random_forest')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with selected features
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    
    # Align indices
    aligned_returns = returns.reindex(features_df.index)
    
    # Drop rows with NaN values
    valid_idx = ~(features_df.isna().any(axis=1) | aligned_returns.isna())
    X = features_df[valid_idx].copy()
    y = aligned_returns[valid_idx].copy()
    
    if X.empty or len(X) < 10:
        print("Not enough valid data for feature selection")
        return features_df  # Return original features
    
    selected_features = []
    
    if method == 'correlation':
        # Calculate correlation with target
        correlations = []
        for col in X.columns:
            try:
                corr = abs(X[col].corr(y))
                correlations.append((col, corr))
            except:
                correlations.append((col, 0))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Get top features
        selected_features = [c[0] for c in correlations[:n_features]]
        
    elif method == 'mutual_info':
        # Use mutual information to select features
        try:
            selector = SelectKBest(mutual_info_regression, k=min(n_features, X.shape[1]))
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = [X.columns[i] for i in selected_indices]
        except Exception as e:
            print(f"Mutual information selection failed: {str(e)}")
            selected_features = X.columns[:n_features]  # Default to first n features
            
    elif method == 'random_forest':
        # Use random forest feature importance
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Train a random forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Select top features
            selected_features = [X.columns[i] for i in indices[:n_features]]
        except Exception as e:
            print(f"Random forest selection failed: {str(e)}")
            selected_features = X.columns[:n_features]  # Default to first n features
    else:
        # Default to first n features
        selected_features = X.columns[:n_features]
    
    # Return selected features
    return features_df[selected_features]

def reduce_dimensions(features_df, n_components=15, method='pca'):
    """
    Reduce dimensionality of features for more efficient LSTM training.
    Uses PCA or a PyTorch autoencoder.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features
    n_components : int
        Number of components to extract
    method : str
        Dimension reduction method ('pca', 'kernel_pca', 'autoencoder')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with reduced dimensions
    """
    from sklearn.decomposition import PCA, KernelPCA
    
    # Drop rows with NaN values
    X = features_df.dropna().copy()
    
    if X.empty or len(X) < 10:
        print("Not enough valid data for dimension reduction")
        return features_df  # Return original features
    
    # Store original index for result
    original_index = X.index
    
    # Standardize features for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimension reduction
    if method == 'pca':
        # Principal Component Analysis
        try:
            pca = PCA(n_components=min(n_components, X.shape[1]))
            X_reduced = pca.fit_transform(X_scaled)
            
            # Create result DataFrame
            result = pd.DataFrame(
                X_reduced, 
                index=original_index,
                columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])]
            )
            
            # Add explained variance as attribute
            result.attrs['explained_variance_ratio'] = pca.explained_variance_ratio_
            result.attrs['components'] = pca.components_
            
            return result
            
        except Exception as e:
            print(f"PCA reduction failed: {str(e)}")
            return features_df  # Return original features
            
    elif method == 'kernel_pca':
        # Kernel PCA for non-linear relationships
        try:
            kpca = KernelPCA(
                n_components=min(n_components, X.shape[1]),
                kernel='rbf',
                gamma=1/X.shape[1]
            )
            X_reduced = kpca.fit_transform(X_scaled)
            
            # Create result DataFrame
            result = pd.DataFrame(
                X_reduced, 
                index=original_index,
                columns=[f'KPC{i+1}' for i in range(X_reduced.shape[1])]
            )
            
            return result
            
        except Exception as e:
            print(f"Kernel PCA reduction failed: {str(e)}")
            return features_df  # Return original features
            
    elif method == 'autoencoder':
        # Use PyTorch autoencoder
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Create a simple autoencoder model
            class Autoencoder(nn.Module):
                def __init__(self, input_dim, encoding_dim):
                    super(Autoencoder, self).__init__()
                    # Encoder
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, encoding_dim*2),
                        nn.ReLU(),
                        nn.Linear(encoding_dim*2, encoding_dim),
                        nn.ReLU()
                    )
                    # Decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, encoding_dim*2),
                        nn.ReLU(),
                        nn.Linear(encoding_dim*2, input_dim)
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return encoded, decoded
            
            # Convert data to tensors
            input_dim = X_scaled.shape[1]
            encoding_dim = min(n_components, input_dim)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Create dataloader
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
            
            # Train autoencoder
            criterion = nn.MSELoss()
            optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
            
            n_epochs = 30
            for epoch in range(n_epochs):
                for data, _ in dataloader:
                    data = data.to(device)
                    
                    # Forward pass
                    encoded, decoded = autoencoder(data)
                    loss = criterion(decoded, data)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Generate encoded features
            autoencoder.eval()
            with torch.no_grad():
                encoded_data, _ = autoencoder(X_tensor.to(device))
                encoded_data = encoded_data.cpu().numpy()
            
            # Create result DataFrame
            result = pd.DataFrame(
                encoded_data, 
                index=original_index,
                columns=[f'AE{i+1}' for i in range(encoded_data.shape[1])]
            )
            
            return result
            
        except Exception as e:
            print(f"Autoencoder reduction failed: {str(e)}")
            return features_df  # Return original features
    else:
        # Return original features if method not recognized
        return features_df


###########################################
# Main Application Functions
###########################################

def generate_features_for_asset(asset_data, asset_class, returns=None):
    """
    Generate appropriate features based on asset class.
    
    Parameters:
    -----------
    asset_data : pd.Series
        Price series for the asset
    asset_class : str
        Asset class ('Commodities', 'Currencies', 'Bonds', 'Equities')
    returns : pd.Series, optional
        Returns series (calculated if not provided)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with asset-specific features
    """
    # Ensure we're working with Series, not DataFrames
    if isinstance(asset_data, pd.DataFrame):
        if 'close' in asset_data.columns:
            asset_data = asset_data['close']
        elif 'value' in asset_data.columns:
            asset_data = asset_data['value']
        else:
            # Use first numeric column
            numeric_cols = asset_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                asset_data = asset_data[numeric_cols[0]]
            else:
                # Fall back to first column
                asset_data = asset_data.iloc[:, 0]
    
    # Calculate returns if not provided
    if returns is None:
        returns = asset_data.pct_change()
    
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        if 'returns' in returns.columns:
            returns = returns['returns']
        else:
            # Use first numeric column
            numeric_cols = returns.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                returns = returns[numeric_cols[0]]
            else:
                # Fall back to first column
                returns = returns.iloc[:, 0]
    
    # Convert to numeric, coercing errors to NaN
    asset_data = pd.to_numeric(asset_data, errors='coerce')
    returns = pd.to_numeric(returns, errors='coerce')
    
    # Generate features based on asset class
    if asset_class == 'Commodities':
        features = calculate_commodity_features(asset_data, returns)
    elif asset_class == 'Currencies':
        features = calculate_fx_features(asset_data, returns)
    elif asset_class == 'Bonds':
        features = calculate_bond_features(asset_data, returns, yield_data=None)
    else:  # Default/Equities
        # Start with common features
        features = pd.DataFrame(index=asset_data.index)
        
        # Add basic statistical features
        stat_features = calculate_statistical_moments(asset_data)
        features = pd.concat([features, stat_features], axis=1)
        
        # Add returns and momentum features
        ret_features = calculate_returns_features(asset_data)
        features = pd.concat([features, ret_features], axis=1)
        
        # Add time series features
        ts_features = calculate_time_series_features(returns)
        features = pd.concat([features, ts_features], axis=1)
        
        # Add volatility and jump features
        vol_features = calculate_volatility_features(returns)
        features = pd.concat([features, vol_features], axis=1)
    
    return features

def generate_features_for_portfolio(prices_dict, asset_class_dict, yield_data=None):
    """
    Generate features for a portfolio of assets.
    
    Parameters:
    -----------
    prices_dict : dict
        Dictionary mapping asset symbols to price series
    asset_class_dict : dict
        Dictionary mapping asset symbols to asset classes
    yield_data : pd.DataFrame, optional
        Yield curve data (for bond assets)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features for all assets
    """
    all_features = {}
    
    # Generate features for each asset
    for symbol, prices in prices_dict.items():
        asset_class = asset_class_dict.get(symbol, 'Equities')  # Default to Equities
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Generate features
        asset_features = generate_features_for_asset(
            asset_data=prices,
            asset_class=asset_class,
            returns=returns,
            yield_data=yield_data
        )
        
        all_features[symbol] = asset_features
    
    return all_features

def create_feature_pipeline(prices, asset_class, normalize=True, select_features=True, 
                          reduce_dims=False, n_features=20, n_components=15):
    """
    Create a complete feature engineering pipeline for a single asset.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series for the asset
    asset_class : str
        Asset class ('Commodities', 'Currencies', 'Bonds', 'Equities')
    normalize : bool
        Whether to normalize features
    select_features : bool
        Whether to select top features
    reduce_dims : bool
        Whether to reduce dimensions
    n_features : int
        Number of top features to select
    n_components : int
        Number of components for dimension reduction
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed features
    """
    # Calculate returns
    returns = prices.pct_change()
    
    # Generate features based on asset class
    features = generate_features_for_asset(prices, asset_class, returns)
    
    # Normalize features if requested
    if normalize:
        if asset_class == 'Commodities':
            # Robust scaling for commodities (fat tails)
            features = normalize_features_for_lstm(features, method='robust')
        elif asset_class == 'Currencies':
            # Standard scaling for FX
            features = normalize_features_for_lstm(features, method='standard')
        elif asset_class == 'Bonds':
            # MinMax scaling for bonds
            features = normalize_features_for_lstm(features, method='minmax')
        else:
            # Quantile transform for equities
            features = normalize_features_for_lstm(features, method='quantile')
    
    # Select top features if requested
    if select_features:
        features = select_top_features(features, returns, n_features=n_features)
    
    # Reduce dimensions if requested
    if reduce_dims:
        features = reduce_dimensions(features, n_components=n_components)
    
    return features

def create_lstm_ready_sequences(features, sequence_length, targets=None, shuffle=True, test_size=0.2):
    """
    Prepare feature sequences for LSTM model training.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame with features
    sequence_length : int
        Length of input sequences
    targets : pd.Series, optional
        Target values (if None, last column of features is used)
    shuffle : bool
        Whether to shuffle the data
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test) arrays
    """
    from sklearn.model_selection import train_test_split
    
    # Drop rows with NaN values
    features_clean = features.dropna()
    
    if features_clean.empty or len(features_clean) <= sequence_length:
        print("Not enough data for sequence creation")
        return None, None, None, None
    
    # Extract targets if not provided
    if targets is None:
        targets = features_clean.iloc[:, -1]
        features_clean = features_clean.iloc[:, :-1]
    else:
        # Align targets with features
        targets = targets.reindex(features_clean.index)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_clean) - sequence_length):
        X.append(features_clean.iloc[i:i+sequence_length].values)
        y.append(targets.iloc[i+sequence_length])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    if shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42)
    else:
        # Time-series split (no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test

def visualize_feature_importance(features, returns, method='correlation'):
    """
    Visualize feature importance based on correlation with returns or other methods.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame with features
    returns : pd.Series
        Returns series
    method : str
        Method for importance calculation ('correlation', 'mutual_info', 'random_forest')
        
    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Align indices
    aligned_returns = returns.reindex(features.index)
    
    # Drop rows with NaN values
    valid_idx = ~(features.isna().any(axis=1) | aligned_returns.isna())
    X = features[valid_idx].copy()
    y = aligned_returns[valid_idx].copy()
    
    if X.empty or len(X) < 10:
        print("Not enough valid data for feature importance visualization")
        return
    
    importances = []
    
    if method == 'correlation':
        # Calculate correlation with target
        for col in X.columns:
            try:
                corr = abs(X[col].corr(y))
                importances.append((col, corr))
            except:
                importances.append((col, 0))
                
    elif method == 'mutual_info':
        # Use mutual information
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi = mutual_info_regression(X, y)
            
            # Normalize to 0-1
            mi_normalized = mi / np.max(mi) if np.max(mi) > 0 else mi
            
            importances = [(col, mi_val) for col, mi_val in zip(X.columns, mi_normalized)]
        except Exception as e:
            print(f"Mutual information calculation failed: {str(e)}")
            return
            
    elif method == 'random_forest':
        # Use random forest feature importance
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Train a random forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            for col, importance in zip(X.columns, rf.feature_importances_):
                importances.append((col, importance))
        except Exception as e:
            print(f"Random forest importance calculation failed: {str(e)}")
            return
    else:
        print(f"Unknown method: {method}")
        return
    
    # Sort importances
    importances.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 20 features
    top_features = importances[:20]
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x=[x[1] for x in top_features], y=[x[0] for x in top_features], orient='h')
    plt.title(f'Feature Importance ({method})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def visualize_pca_components(features, n_components=3):
    """
    Visualize PCA components and explained variance.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame with features
    n_components : int
        Number of components to visualize
        
    Returns:
    --------
    None
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Drop rows with NaN values
    X = features.dropna().copy()
    
    if X.empty or len(X) < 10:
        print("Not enough valid data for PCA visualization")
        return
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            np.cumsum(pca.explained_variance_ratio_), 'r-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.tight_layout()
    plt.show()
    
    # Plot feature loadings on first two components
    if X.shape[1] > 2:
        plt.figure(figsize=(12, 10))
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            index=X.columns
        )
        
        # Select top features by loading magnitude
        loadings['magnitude'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
        top_features = loadings.nlargest(15, 'magnitude').index
        
        # Create scatter plot of loadings
        plt.figure(figsize=(12, 10))
        plt.scatter(loadings.loc[top_features, 'PC1'], loadings.loc[top_features, 'PC2'])
        
        # Add feature labels
        for feature in top_features:
            plt.annotate(feature, (loadings.loc[feature, 'PC1'], loadings.loc[feature, 'PC2']))
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Feature Loadings')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()