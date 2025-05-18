"""
Data utilities for financial time series loading and preprocessing.
Compatible with CudaOptimizedTrendPredictor and existing AlphaVantage client.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpha_vantage_client_optimized import AlphaVantageClient
from sklearn.preprocessing import MinMaxScaler
import pickle

def load_financial_data(symbols, start_date=None, end_date=None, data_path='./data', 
                       api_key=None, use_cache=True, cache_days=1):
    """
    Load financial data for given symbols using AlphaVantage API or cached data.
    
    Parameters:
    -----------
    symbols : list or str
        List of ticker symbols or comma-separated string
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format (defaults to current date)
    data_path : str, optional
        Path to data directory
    api_key : str, optional
        AlphaVantage API key (if not provided, will look for ALPHA_VANTAGE_API_KEY env var)
    use_cache : bool, optional
        Whether to use cached data if available
    cache_days : int, optional
        Number of days before cache expires
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with financial data for all symbols
    """
    # Convert string to list if needed
    if isinstance(symbols, str):
        symbols = symbols.split(',')
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 5 years of data
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Cache file path
    cache_file = os.path.join(data_path, f"financial_data_{'_'.join(symbols)}.pkl")
    
    # Check cache first if enabled
    if use_cache and os.path.exists(cache_file):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_age.days < cache_days:
            print(f"Loading data from cache ({cache_age.days} days old)")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                
            # Filter data by date
            data = data.loc[start_date:end_date]
            return data
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if api_key is None:
            raise ValueError("No AlphaVantage API key provided. Set ALPHA_VANTAGE_API_KEY env var.")
    
    # Initialize AlphaVantage client
    av_client = AlphaVantageClient(api_key)
    
    # Initialize data dictionary
    data_dict = {}
    
    # Load data for each symbol
    for symbol in symbols:
        try:
            # Get daily adjusted data
            symbol_data = av_client.get_daily_adjusted(symbol, outputsize="full", cache=True)
            
            # Add to data dictionary
            if not symbol_data.empty:
                data_dict[symbol] = symbol_data
                print(f"Loaded data for {symbol} with {len(symbol_data)} records")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error loading data for {symbol}: {str(e)}")
    
    # Combine data for all symbols
    if not data_dict:
        raise ValueError("No data loaded for any symbols")
    
    # Combine all dataframes into one
    combined_data = pd.concat(data_dict, axis=1)
    
    # Ensure hierarchical column names
    if not isinstance(combined_data.columns, pd.MultiIndex):
        # Restructure the dataframe to have proper MultiIndex
        new_data = {}
        for symbol, df in data_dict.items():
            for col in df.columns:
                new_data[(symbol, col)] = df[col]
        
        combined_data = pd.DataFrame(new_data)
    
    # Filter by date
    combined_data = combined_data.loc[start_date:end_date]
    
    # Cache the data
    with open(cache_file, 'wb') as f:
        pickle.dump(combined_data, f)
    
    return combined_data

def preprocess_data(data, sequence_length=60, train_ratio=0.8, feature_columns=None, 
                   target_column='adjusted_close', scale_data=True):
    """
    Preprocess financial data for model training.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Financial data with MultiIndex columns (symbol, feature)
    sequence_length : int, optional
        Length of sequences for LSTM input
    train_ratio : float, optional
        Ratio of data to use for training (0.0-1.0)
    feature_columns : list, optional
        List of column names to use as features (defaults to close, high, low, volume)
    target_column : str, optional
        Column to use as target (default: adjusted_close)
    scale_data : bool, optional
        Whether to scale the data to [-1, 1] range
        
    Returns:
    --------
    tuple
        (train_data, test_data) where each is a dict with 'X' and 'y' keys
    """
    if feature_columns is None:
        feature_columns = ['close', 'high', 'low', 'volume']
    
    # Get all symbols from data
    if isinstance(data.columns, pd.MultiIndex):
        symbols = data.columns.levels[0]
    else:
        # Assume single symbol data
        symbols = [data.name] if hasattr(data, 'name') else ['unknown']
        # Restructure to match expected format
        data = pd.concat({symbols[0]: data}, axis=1)
    
    # Dictionary to store processed data
    train_data = {'X': [], 'y': [], 'symbols': []}
    test_data = {'X': [], 'y': [], 'symbols': []}
    
    # Store scalers for inverse transformation
    scalers = {}
    
    # Process each symbol
    for symbol in symbols:
        try:
            # Extract features for this symbol
            symbol_features = []
            for feature in feature_columns:
                if (symbol, feature) in data.columns:
                    symbol_features.append(data[(symbol, feature)])
            
            # Extract target
            if (symbol, target_column) in data.columns:
                symbol_target = data[(symbol, target_column)]
            elif (symbol, 'close') in data.columns:  # fallback to close if adjusted_close not available
                symbol_target = data[(symbol, 'close')]
            else:
                print(f"Target column not found for {symbol}, skipping")
                continue
            
            # Combine features
            X = pd.concat(symbol_features, axis=1)
            X.columns = feature_columns
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            y = symbol_target.fillna(method='ffill').fillna(method='bfill')
            
            # Scale data if requested
            if scale_data:
                feature_scaler = MinMaxScaler(feature_range=(-1, 1))
                target_scaler = MinMaxScaler(feature_range=(-1, 1))
                
                X_scaled = feature_scaler.fit_transform(X)
                y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
                
                # Store scalers
                scalers[symbol] = {
                    'feature': feature_scaler,
                    'target': target_scaler
                }
            else:
                X_scaled = X.values
                y_scaled = y.values
            
            # Create sequences
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - sequence_length):
                X_seq.append(X_scaled[i:i+sequence_length])
                y_seq.append(y_scaled[i+sequence_length])
            
            if not X_seq:
                print(f"No sequences created for {symbol}, skipping")
                continue
                
            # Convert to numpy arrays
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Split into train and test
            split_idx = int(len(X_seq) * train_ratio)
            
            # Add to train data
            train_data['X'].append(X_seq[:split_idx])
            train_data['y'].append(y_seq[:split_idx])
            train_data['symbols'].extend([symbol] * split_idx)
            
            # Add to test data
            test_data['X'].append(X_seq[split_idx:])
            test_data['y'].append(y_seq[split_idx:])
            test_data['symbols'].extend([symbol] * (len(X_seq) - split_idx))
            
        except Exception as e:
            print(f"Error preprocessing data for {symbol}: {str(e)}")
    
    # Combine data from all symbols
    train_data['X'] = np.vstack(train_data['X']) if train_data['X'] else np.array([])
    train_data['y'] = np.concatenate(train_data['y']) if train_data['y'] else np.array([])
    
    test_data['X'] = np.vstack(test_data['X']) if test_data['X'] else np.array([])
    test_data['y'] = np.concatenate(test_data['y']) if test_data['y'] else np.array([])
    
    # Add scalers to the data dictionaries
    train_data['scalers'] = scalers
    test_data['scalers'] = scalers
    
    return train_data, test_data