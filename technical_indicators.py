# technical_indicators.py

import numpy as np
import pandas as pd

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return pd.DataFrame({'RSI': rsi})

def calculate_moving_averages(data, short_window=20, long_window=50):
    """Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)."""
    sma = pd.DataFrame({'SMA': data.rolling(window=short_window).mean()})
    ema = pd.DataFrame({'EMA': data.ewm(span=short_window, adjust=False).mean()})
    
    return sma, ema

def calculate_stochastic(data, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator."""
    low_min = data.rolling(window=k_window).min()
    high_max = data.rolling(window=k_window).max()
    
    k = 100 * ((data - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    
    return pd.DataFrame({'SlowK': k, 'SlowD': d})

def calculate_adx(high, low, close, window=14):
    """Calculate Average Directional Index (ADX)."""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'TR': pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)})
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pos_dm = pd.Series(0.0, index=up_move.index)
    neg_dm = pd.Series(0.0, index=down_move.index)
    
    pos_dm[(up_move > down_move) & (up_move > 0)] = up_move
    neg_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # Indicators
    atr = tr['TR'].ewm(span=window, adjust=False).mean()
    plus_di = 100 * (pos_dm.ewm(span=window, adjust=False).mean() / atr)
    minus_di = 100 * (neg_dm.ewm(span=window, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=window, adjust=False).mean()
    
    return pd.DataFrame({'ADX': adx, 'PlusDI': plus_di, 'MinusDI': minus_di})