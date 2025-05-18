# enhanced_feature_example.py
"""
Example of using enhanced feature engineering with CudaOptimizedTrendPredictor.
This script demonstrates how to apply the advanced features from the research paper
to generate specialized features for commodities, FX pairs, and bonds.
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Import the predictor and enhancement components
from cuda_optimized_trend_predictor import CudaOptimizedTrendPredictor
from feature_engineering_integration import FeatureEngineeringEnhancer
from advanced_features import (
    calculate_commodity_features,
    calculate_fx_features,
    calculate_bond_features,
    visualize_feature_importance
)
from config import ASSETS, MODEL_PARAMS, CUDA_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_predictor():
    """
    Set up the CudaOptimizedTrendPredictor with API key from environment.
    
    Returns:
    --------
    CudaOptimizedTrendPredictor
        Initialized predictor
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
    
    # Initialize predictor
    predictor = CudaOptimizedTrendPredictor(
        api_key=api_key,
        assets_dict=ASSETS,
        model_params=MODEL_PARAMS,
        cuda_settings=CUDA_SETTINGS
    )
    
    return predictor

def show_feature_examples(predictor):
    """
    Show examples of advanced features for different asset classes.
    
    Parameters:
    -----------
    predictor : CudaOptimizedTrendPredictor
        Trend predictor instance
        
    Returns:
    --------
    None
    """
    logger.info("Generating feature examples for different asset classes...")
    
    # Dictionary to store example data
    examples = {}
    
    # Get sample data for each asset class
    for asset_class, tickers in predictor.assets_dict.items():
        if not tickers:
            continue
            
        # Get the first ticker from each asset class
        ticker = tickers[0]
        logger.info(f"Getting data for {ticker} ({asset_class})")
        
        try:
            # Get data using AlphaVantage client
            if asset_class == 'Commodities':
                # Use commodity data method
                data = predictor.commodity_client.get_commodity_data(ticker)
                # Extract price column
                if 'close' in data.columns:
                    price_data = data['close']
                elif 'value' in data.columns:
                    price_data = data['value']
                else:
                    price_data = data.iloc[:, 0]  # Use first column
                    
            elif asset_class == 'Currencies':
                # For FX pairs
                if '/' in ticker:
                    from_currency, to_currency = ticker.split('/')
                    data = predictor.av_client.get_fx_rate(from_currency, to_currency)
                else:
                    # Default to USD as quote currency
                    data = predictor.av_client.get_fx_rate(ticker, 'USD')
                price_data = data['close']
                
            elif asset_class == 'Bonds':
                # For bonds
                data = predictor.av_client.get_daily_adjusted(ticker)
                price_data = data['close']
                
            else:
                # For equities
                data = predictor.av_client.get_daily_adjusted(ticker)
                price_data = data['close']
            
            # Calculate returns
            returns = price_data.pct_change()
            
            # Store data
            examples[asset_class] = {
                'ticker': ticker,
                'price_data': price_data,
                'returns': returns,
                'raw_data': data
            }
            
            logger.info(f"Successfully retrieved data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error getting data for {ticker}: {str(e)}")
    
    # Generate features for each asset class
    for asset_class, data in examples.items():
        ticker = data['ticker']
        price_data = data['price_data']
        returns = data['returns']
        
        logger.info(f"Generating features for {ticker} ({asset_class})")
        
        try:
            # Generate features based on asset class
            if asset_class == 'Commodities':
                features = calculate_commodity_features(price_data, returns)
                
                # Show an example of commodity-specific features
                print(f"\nCommodody-specific features for {ticker}:")
                if 'seasonal' in features.columns:
                    print("\nSeasonal decomposition features:")
                    print(features[['seasonal', 'trend', 'residual']].describe())
                
                # Show top jump features
                if 'jump_indicator' in features.columns:
                    jump_count = features['jump_indicator'].sum()
                    print(f"\nDetected {jump_count} price jumps")
                
                # Show wavelet features
                wavelet_cols = [col for col in features.columns if 'wavelet' in col]
                if wavelet_cols:
                    print("\nWavelet transform features:")
                    print(features[wavelet_cols[:3]].describe())
                
                # Plot some commodity features
                plt.figure(figsize=(12, 10))
                
                # Plot 1: Price and trend
                plt.subplot(3, 1, 1)
                if 'trend' in features.columns:
                    plt.plot(price_data.index, price_data, label='Price')
                    plt.plot(features.index, features['trend'], label='Trend')
                    plt.title(f"{ticker} Price and Trend Component")
                    plt.legend()
                else:
                    plt.plot(price_data.index, price_data, label='Price')
                    plt.title(f"{ticker} Price")
                
                # Plot 2: Seasonal component
                plt.subplot(3, 1, 2)
                if 'seasonal' in features.columns:
                    plt.plot(features.index, features['seasonal'])
                    plt.title("Seasonal Component")
                elif 'wavelet_approx' in features.columns:
                    plt.plot(features.index, features['wavelet_approx'])
                    plt.title("Wavelet Approximation")
                
                # Plot 3: Detected jumps
                plt.subplot(3, 1, 3)
                if 'jump_size' in features.columns:
                    plt.scatter(features.index, features['jump_size'], color='red', alpha=0.7)
                    plt.title("Detected Price Jumps")
                elif 'rv_20d' in features.columns:
                    plt.plot(features.index, features['rv_20d'])
                    plt.title("Realized Volatility (20-day)")
                
                plt.tight_layout()
                plt.show()
                
            elif asset_class == 'Currencies':
                features = calculate_fx_features(price_data, returns)
                
                # Show an example of FX-specific features
                print(f"\nFX-specific features for {ticker}:")
                
                # Show autocorrelation features
                acf_cols = [col for col in features.columns if 'acf_lag' in col]
                if acf_cols:
                    print("\nAutocorrelation features:")
                    print(features[acf_cols[:3]].describe())
                
                # Show Hurst exponent
                hurst_cols = [col for col in features.columns if 'hurst' in col]
                if hurst_cols:
                    print("\nHurst exponent features (trend vs. mean-reversion):")
                    print(features[hurst_cols].describe())
                    
                    # Interpret Hurst values
                    avg_hurst = features[hurst_cols[0]].mean()
                    if avg_hurst > 0.55:
                        trend_type = "trending"
                    elif avg_hurst < 0.45:
                        trend_type = "mean-reverting"
                    else:
                        trend_type = "random walk"
                    print(f"Average Hurst exponent: {avg_hurst:.3f} ({trend_type})")
                
                # Plot some FX features
                plt.figure(figsize=(12, 10))
                
                # Plot 1: Price and momentum
                plt.subplot(3, 1, 1)
                plt.plot(price_data.index, price_data, label='Price')
                if 'momentum_20d' in features.columns:
                    # Normalize for plotting
                    norm_momentum = features['momentum_20d'] / features['momentum_20d'].abs().max() * price_data.max() * 0.3
                    plt.plot(features.index, norm_momentum + price_data.min(), label='Momentum (20d)')
                plt.title(f"{ticker} Price and Momentum")
                plt.legend()
                
                # Plot 2: Autocorrelation
                plt.subplot(3, 1, 2)
                if acf_cols and len(acf_cols) >= 3:
                    for col in acf_cols[:3]:
                        plt.plot(features.index, features[col], label=col)
                    plt.title("Autocorrelation Features")
                    plt.legend()
                elif 'rv_20d' in features.columns:
                    plt.plot(features.index, features['rv_20d'])
                    plt.title("Realized Volatility (20-day)")
                
                # Plot 3: Hurst exponent
                plt.subplot(3, 1, 3)
                if hurst_cols:
                    plt.plot(features.index, features[hurst_cols[0]])
                    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                    plt.fill_between(features.index, 0.45, 0.55, alpha=0.2, color='gray')
                    plt.title("Hurst Exponent (>0.5: Trending, <0.5: Mean-reverting)")
                
                plt.tight_layout()
                plt.show()
                
            elif asset_class == 'Bonds':
                features = calculate_bond_features(price_data, returns)
                
                # Show an example of bond-specific features
                print(f"\nBond-specific features for {ticker}:")
                
                # Show RSI features
                rsi_cols = [col for col in features.columns if 'rsi_' in col]
                if rsi_cols:
                    print("\nRSI features:")
                    print(features[rsi_cols[:3]].describe())
                
                # Show bond momentum features
                momentum_cols = [col for col in features.columns if 'ma_cross' in col or 'price_vs_ma' in col]
                if momentum_cols:
                    print("\nBond momentum features:")
                    print(features[momentum_cols[:3]].describe())
                
                # Plot some bond features
                plt.figure(figsize=(12, 10))
                
                # Plot 1: Price and moving averages
                plt.subplot(3, 1, 1)
                plt.plot(price_data.index, price_data, label='Price')
                for ma in ['ma_20', 'ma_60']:
                    if ma in features.columns:
                        plt.plot(features.index, features[ma], label=f"{ma.replace('ma_', 'SMA-')}")
                plt.title(f"{ticker} Price and Moving Averages")
                plt.legend()
                
                # Plot 2: RSI indicator
                plt.subplot(3, 1, 2)
                if 'rsi_14' in features.columns:
                    plt.plot(features.index, features['rsi_14'])
                    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)
                    plt.axhline(y=30, color='g', linestyle='--', alpha=0.7)
                    plt.title("RSI (14-day)")
                    plt.ylim(0, 100)
                elif 'rv_20d' in features.columns:
                    plt.plot(features.index, features['rv_20d'])
                    plt.title("Realized Volatility (20-day)")
                
                # Plot 3: Price relative to moving average
                plt.subplot(3, 1, 3)
                price_vs_ma_cols = [col for col in features.columns if 'price_vs_ma' in col]
                if price_vs_ma_cols:
                    for col in price_vs_ma_cols[:2]:
                        plt.plot(features.index, features[col] * 100, label=col)
                    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    plt.title("Price vs. Moving Average (%)")
                    plt.legend()
                
                plt.tight_layout()
                plt.show()
                
            else:  # Equities
                # Calculate standard features
                from advanced_features import (
                    calculate_statistical_moments,
                    calculate_returns_features,
                    calculate_volatility_features,
                    calculate_time_series_features
                )
                
                # Generate features
                stat_features = calculate_statistical_moments(price_data)
                ret_features = calculate_returns_features(price_data)
                vol_features = calculate_volatility_features(returns)
                ts_features = calculate_time_series_features(returns)
                
                # Combine features
                features = pd.concat([stat_features, ret_features, vol_features, ts_features], axis=1)
                
                # Show an example of equity features
                print(f"\nStandard features for {ticker} (Equities):")
                
                # Show statistical moments
                print("\nStatistical moments:")
                stat_cols = [col for col in features.columns if 'mean_' in col or 'variance_' in col]
                if stat_cols:
                    print(features[stat_cols[:3]].describe())
                
                # Show volatility features
                vol_cols = [col for col in features.columns if 'std_' in col or 'rv_' in col]
                if vol_cols:
                    print("\nVolatility features:")
                    print(features[vol_cols[:3]].describe())
                
                # Plot some equity features
                plt.figure(figsize=(12, 10))
                
                # Plot 1: Price and Z-score
                plt.subplot(3, 1, 1)
                plt.plot(price_data.index, price_data, label='Price')
                zscore_cols = [col for col in features.columns if 'zscore_' in col]
                if zscore_cols:
                    # Normalize for plotting
                    norm_zscore = features[zscore_cols[0]] * price_data.std() + price_data.mean()
                    plt.plot(features.index, norm_zscore, label='Z-score', alpha=0.7)
                plt.title(f"{ticker} Price and Z-score")
                plt.legend()
                
                # Plot 2: Volatility
                plt.subplot(3, 1, 2)
                for col in vol_cols[:2]:
                    if col in features.columns:
                        plt.plot(features.index, features[col], label=col)
                plt.title("Volatility Features")
                plt.legend()
                
                # Plot 3: Returns
                plt.subplot(3, 1, 3)
                ret_cols = [col for col in features.columns if 'return_' in col and 'd' in col]
                if ret_cols:
                    for col in ret_cols[:2]:
                        plt.plot(features.index, features[col], label=col)
                    plt.title("Return Features")
                    plt.legend()
                
                plt.tight_layout()
                plt.show()
            
            # Visualize feature importance
            try:
                plt.figure(figsize=(14, 10))
                visualize_feature_importance(features, returns)
                plt.title(f"Feature Importance for {ticker}")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.error(f"Error visualizing feature importance: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error generating features for {ticker}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

def enhance_predictor_example():
    """
    Full example of enhancing a predictor with advanced features.
    
    Returns:
    --------
    CudaOptimizedTrendPredictor
        Enhanced predictor
    """
    # Set up predictor
    predictor = setup_predictor()
    
    # Load data
    logger.info("Loading returns data...")
    predictor.load_returns_data()
    
    # Instead of loading technical indicators the old way (which would fail
    # for commodities, FX, and bonds), use our new approach
    logger.info("Generating advanced features...")
    
    # Create feature enhancer
    enhancer = FeatureEngineeringEnhancer(
        predictor=predictor,
        n_features=20,        # Number of top features to select per asset
        normalize_method="asset_class"  # Different normalization per asset class
    )
    
    # Enhance predictor with advanced features
    success = enhancer.enhance_predictor()
    
    if success:
        logger.info("Successfully enhanced predictor with advanced features")
        
        # Continue with model building as before
        if not os.path.exists('models') or len(os.listdir('models')) == 0:
            logger.info("Building trend models...")
            predictor.build_trend_models()
            predictor.save_trend_models()
        else:
            logger.info("Loading existing trend models...")
            predictor.load_trend_models()
        
        # Generate predictions
        logger.info("Generating trend predictions...")
        predictor.predict_trends()
        
        # Adjust for economic conditions
        logger.info("Adjusting for economic conditions...")
        predictor.adjust_for_economic_uncertainty()
        
        # Adjust for news sentiment
        logger.info("Adjusting for news sentiment...")
        predictor.adjust_for_news_sentiment()
    else:
        logger.error("Failed to enhance predictor with advanced features")
    
    return predictor

def simple_usage_example():
    """
    Simple example of using the patched trend predictor.
    This uses the method added to CudaOptimizedTrendPredictor by the patch.
    
    Returns:
    --------
    None
    """
    # Set up predictor
    predictor = setup_predictor()
    
    # Load data
    logger.info("Loading returns data...")
    predictor.load_returns_data()
    
    # Use the patched method to enhance the predictor
    logger.info("Enhancing predictor with advanced features...")
    predictor.enhance_with_advanced_features(
        n_features=20, 
        normalize_method="asset_class"
    )
    
    # Continue with model building and prediction as before
    logger.info("Building models and generating predictions...")
    
    # Use existing models or build new ones
    if not os.path.exists('models') or len(os.listdir('models')) == 0:
        predictor.build_trend_models()
        predictor.save_trend_models()
    else:
        predictor.load_trend_models()
    
    # Generate predictions
    predictor.predict_trends()
    predictor.adjust_for_economic_uncertainty()
    predictor.adjust_for_news_sentiment()
    
    # Create allocation
    from allocation_utils import create_optimized_etf_allocation
    
    logger.info("Creating optimized ETF allocation...")
    allocation = create_optimized_etf_allocation(
        predictor=predictor,
        risk_target=0.15,   # 15% annualized volatility
        max_leverage=2.0    # 2x leverage
    )
    
    # Display allocation
    print("\nOptimized ETF Allocation:")
    pd.set_option('display.float_format', '{:.2%}'.format)
    print(allocation[['Position', 'Volatility', 'Asset_Class']])
    pd.reset_option('display.float_format')
    
    # Visualize allocation
    from visualization_utils import visualize_etf_allocation, plot_risk_contributions
    
    visualize_etf_allocation(allocation)
    plot_risk_contributions(allocation)
    
    return predictor, allocation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced feature engineering example")
    parser.add_argument("--show-examples", action="store_true", help="Show feature examples for different asset classes")
    parser.add_argument("--full-example", action="store_true", help="Run the full enhancement example")
    parser.add_argument("--simple-example", action="store_true", help="Run the simple usage example")
    
    args = parser.parse_args()
    
    # Default to simple example if no arguments provided
    if not (args.show_examples or args.full_example or args.simple_example):
        args.simple_example = True
    
    if args.show_examples:
        predictor = setup_predictor()
        predictor.load_returns_data()
        show_feature_examples(predictor)
    
    if args.full_example:
        enhance_predictor_example()
    
    if args.simple_example:
        simple_usage_example()