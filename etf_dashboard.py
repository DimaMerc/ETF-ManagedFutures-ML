# etf_dashboard.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
from cuda_optimized_trend_predictor import PytorchTrendPredictor
from config import ASSETS, MODEL_PARAMS, PORTFOLIO_SETTINGS
from dotenv import load_dotenv
import argparse


def create_etf_allocation(predictor, risk_target=0.10, max_leverage=1.5, 
                          min_signal_strength=0.0005, # Minimum signal strength threshold
                          max_class_risk=0.50,        # Reduced from 0.60
                          min_position_size=0.005):   # Minimum position size (0.5%)
    """
    Create ETF allocation with balanced risk across asset classes.
    
     Parameters:
    -----------
    predictor : PytorchTrendPredictor
        Trend predictor instance with predictions
    risk_target : float, optional
        Target volatility (default: 0.10)
    max_leverage : float, optional
        Maximum leverage allowed (default: 1.5)
    min_signal_strength : float, optional
        Minimum absolute signal strength to take a position
    max_class_risk : float, optional
        Maximum risk contribution from a single asset class
    min_position_size : float, optional
        Minimum absolute position size (as percentage)
        
    Returns:
    --------
    pd.DataFrame
        ETF allocation
    """
    print(f"Creating ETF allocation with min_signal_strength={min_signal_strength}, "
          f"max_class_risk={max_class_risk}, min_position_size={min_position_size}")
    
    # Check if we have any predictions
    has_predictions = (hasattr(predictor, 'final_direction') and 
                      hasattr(predictor, 'final_strength') and
                      predictor.final_direction is not None and
                      predictor.final_strength is not None)
    
    if not has_predictions:
        has_predictions = (hasattr(predictor, 'adjusted_direction') and 
                          hasattr(predictor, 'adjusted_strength') and
                          predictor.adjusted_direction is not None and
                          predictor.adjusted_strength is not None)
    
    if not has_predictions:
        has_predictions = (hasattr(predictor, 'trend_direction') and 
                          hasattr(predictor, 'trend_strength') and
                          predictor.trend_direction is not None and
                          predictor.trend_strength is not None)
    
    if not has_predictions:
        print("No predictions available to create allocation.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Direction', 'Strength', 'Volatility', 
                                   'Position', 'Allocation', 'Asset_Class'])
    
    # Get predictions (use final if available, otherwise adjusted, otherwise trend)
    if hasattr(predictor, 'final_direction') and hasattr(predictor, 'final_strength'):
        predictions_df = pd.DataFrame({
            'Direction': predictor.final_direction,
            'Strength': predictor.final_strength
        })
    elif hasattr(predictor, 'adjusted_direction') and hasattr(predictor, 'adjusted_strength'):
        predictions_df = pd.DataFrame({
            'Direction': predictor.adjusted_direction,
            'Strength': predictor.adjusted_strength
        })
    else:
        predictions_df = pd.DataFrame({
            'Direction': predictor.trend_direction,
            'Strength': predictor.trend_strength
        })
    
    # Calculate historical volatilities using returns data
    if predictor.returns is not None:
        volatilities = predictor.returns.std() * np.sqrt(252)  # Annualized
        predictions_df['Volatility'] = volatilities
    else:
        # Default volatility if no returns data
        predictions_df['Volatility'] = 0.20
    
    # Add asset class information
    predictions_df['Asset_Class'] = 'Unknown'
    
    # Define asset classes
    asset_class_groups = {
        'Commodities': ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE'],
        'Currencies': ['EUR', 'GBP', 'JPY', 'CAD', 'AUD'],
        'Bonds': ['TREASURY_YIELD'],
        'Equities': ['SPY', 'QQQ', 'DIA', 'IWM']
    }
    
    for asset_class, tickers in asset_class_groups.items():
        for ticker in tickers:
            if ticker in predictions_df.index:
                predictions_df.loc[ticker, 'Asset_Class'] = asset_class
    
    # Calculate raw position weights
    predictions_df['Raw_Weight'] = predictions_df['Direction'] * predictions_df['Strength']
    
    # Apply minimum signal strength filter
    abs_weights = predictions_df['Raw_Weight'].abs()
    predictions_df.loc[abs_weights < min_signal_strength, 'Raw_Weight'] = 0
    predictions_df.loc[abs_weights < min_signal_strength, 'Direction'] = 0
    
    # Risk-adjusted position sizing
    predictions_df['Risk_Contribution'] = predictions_df['Raw_Weight'] * predictions_df['Volatility']
    
    # Calculate risk contribution by asset class
    asset_class_risk = predictions_df.groupby('Asset_Class')['Risk_Contribution'].sum().abs()
    total_risk = asset_class_risk.sum()
    
    # Apply asset class risk caps if needed
    risk_scale_factors = {}
    for asset_class, risk in asset_class_risk.items():
        if risk / total_risk > max_class_risk:
            print(f"Asset class {asset_class} exceeds max risk: {risk/total_risk:.1%} > {max_class_risk:.1%}")
            risk_scale_factors[asset_class] = (max_class_risk * total_risk) / risk
        else:
            risk_scale_factors[asset_class] = 1.0
    
    # Scale risk contributions by asset class
    for asset_class, scale_factor in risk_scale_factors.items():
        mask = predictions_df['Asset_Class'] == asset_class
        predictions_df.loc[mask, 'Risk_Contribution'] *= scale_factor
    
    # Normalize weights to meet target volatility
    # Calculate portfolio volatility and scale accordingly
    if predictions_df['Risk_Contribution'].sum() != 0:
        # First calculate what portfolio volatility would be without scaling
        portfolio_vol = np.sqrt((predictions_df['Risk_Contribution'] ** 2).sum())
        
        # If portfolio volatility is too low, scale up positions
        if portfolio_vol > 0:
            # Scale factor to reach target volatility
            vol_scale_factor = risk_target / portfolio_vol
            print(f"Portfolio volatility scaling: {portfolio_vol:.2%} -> {risk_target:.2%} (factor: {vol_scale_factor:.2f}x)")
            
            # Scale positions to reach target volatility
            predictions_df['Position'] = predictions_df['Risk_Contribution'] * vol_scale_factor
        else:
            predictions_df['Position'] = 0
    else:
        predictions_df['Position'] = 0
    
    # Apply minimum position size (remove tiny positions)
    abs_positions = predictions_df['Position'].abs()
    small_positions = abs_positions < min_position_size
    
    # Print positions being filtered out
    if small_positions.any():
        print(f"Removing {small_positions.sum()} positions below minimum size of {min_position_size:.2%}")
        predictions_df.loc[small_positions, 'Position'] = 0
    
    # Apply leverage constraint
    total_leverage = predictions_df['Position'].abs().sum()
    if total_leverage > max_leverage:
        predictions_df['Position'] = predictions_df['Position'] * (max_leverage / total_leverage)
        print(f"Reducing leverage from {total_leverage:.2f}x to {max_leverage:.2f}x")
    
    # Calculate dollar allocation (assuming $1M portfolio)
    portfolio_value = 1000000
    predictions_df['Allocation'] = predictions_df['Position'] * portfolio_value
    
    # Calculate final portfolio volatility
    final_vol = np.sqrt((predictions_df['Position'] * predictions_df['Volatility']) ** 2).sum()
    print(f"Final portfolio volatility: {final_vol:.2%}")
    
    # Sort by position size (descending by absolute value)
    predictions_df = predictions_df.sort_values('Position', key=abs, ascending=False)
    
    # Calculate and print statistics
    long_exposure = predictions_df[predictions_df['Position'] > 0]['Position'].sum()
    short_exposure = predictions_df[predictions_df['Position'] < 0]['Position'].sum()
    
    print(f"Long exposure: {long_exposure:.2%}")
    print(f"Short exposure: {short_exposure:.2%}")
    print(f"Net exposure: {long_exposure + short_exposure:.2%}")
    
    return predictions_df

def visualize_etf_allocation(allocation_df):
    """
    Visualize ETF allocation.
    
    Parameters:
    -----------
    allocation_df : pd.DataFrame
        ETF allocation dataframe
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(12, 8))
    
    # Create bar plot of positions
    positions = allocation_df['Position']
    colors = ['green' if x > 0 else 'red' for x in positions]
    
    plt.barh(allocation_df.index, positions, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title('Managed Futures ETF Allocation')
    plt.xlabel('Position Weight')
    plt.ylabel('Asset')
    plt.grid(axis='x', alpha=0.3)
    
    # Add leverage information
    total_leverage = allocation_df['Position'].abs().sum()
    plt.figtext(0.5, 0.01, f'Total Leverage: {total_leverage:.2f}x', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_risk_contributions(allocation_df):
    """
    Plot risk contributions by asset class.
    
    Parameters:
    -----------
    allocation_df : pd.DataFrame
        ETF allocation dataframe
    
    Returns:
    --------
    None
    """
    # Add asset class
    allocation_df['Asset_Class'] = 'Unknown'
    
    for asset_class, tickers in ASSETS.items():
        for ticker in tickers:
            if ticker in allocation_df.index:
                allocation_df.loc[ticker, 'Asset_Class'] = asset_class
    
    # Calculate risk contribution
    allocation_df['Risk_Contribution_Actual'] = allocation_df['Position'] * allocation_df['Volatility']
    
    # Group by asset class
    risk_by_class = allocation_df.groupby('Asset_Class')['Risk_Contribution_Actual'].sum()
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(risk_by_class.abs(), labels=risk_by_class.index, 
            autopct='%1.1f%%', startangle=90, 
            colors=plt.cm.tab10.colors[:len(risk_by_class)])
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title('Risk Contribution by Asset Class')
    plt.tight_layout()
    plt.show()

def dashboard(build_models=False):
    """
    Create an ETF dashboard.
    """
    # Load API key from environment variable
    load_dotenv()

    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        print("Warning: No AlphaVantage API key found in environment variables.")
        print("Set the ALPHA_VANTAGE_API_KEY environment variable.")
        return
    
    # Set PyTorch configurations
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Using {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    print("\n=== Managed Futures ETF Dashboard ===\n")
    
    # Initialize predictor
    predictor = PytorchTrendPredictor(
        api_key=api_key,
        assets_dict=ASSETS,
        sequence_length=MODEL_PARAMS['sequence_length'],
        epochs=MODEL_PARAMS['epochs']
    )
    
    # Load data and generate predictions
    print("Loading returns data...")
    predictor.load_returns_data()
    
    print("Loading technical indicators...")
    predictor.load_technical_indicators()
    
    print("Loading economic data...")
    predictor.load_economic_data()
    
    print("Loading news sentiment...")
    predictor.load_news_sentiment()
    
    if build_models or not os.path.exists('models') or len(os.listdir('models')) == 0:
        print("Building new trend models...")
        predictor.build_trend_models()
        predictor.save_trend_models()
    else:
        print("Loading existing trend models...")
        predictor.load_trend_models()
    
    # Generate predictions
    print("Generating trend predictions...")
    predictor.predict_trends()
    
    print("Adjusting for economic conditions...")
    predictor.adjust_for_economic_uncertainty()
    
    print("Adjusting for news sentiment...")
    predictor.adjust_for_news_sentiment()
    
    # Create ETF allocation
    allocation_df = create_etf_allocation(
        predictor,
        risk_target=PORTFOLIO_SETTINGS['risk_target'],
        max_leverage=PORTFOLIO_SETTINGS['max_leverage']
    )
    
    # Display allocation
    print("\n=== ETF Allocation ===\n")
    pd.set_option('display.float_format', '{:.2%}'.format)
    print(allocation_df[['Position', 'Volatility']])
    pd.reset_option('display.float_format')
    
    # Calculate portfolio statistics
    total_leverage = allocation_df['Position'].abs().sum()
    net_exposure = allocation_df['Position'].sum()
    weighted_volatility = (allocation_df['Position'] * allocation_df['Volatility']).sum()
    
    print(f"\nTotal Leverage: {total_leverage:.2f}x")
    print(f"Net Exposure: {net_exposure:.2%}")
    print(f"Portfolio Volatility: {weighted_volatility:.2%}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_etf_allocation(allocation_df)
    plot_risk_contributions(allocation_df)
    
    # Display top positions
    print("\n=== Top Long Positions ===")
    longs = allocation_df[allocation_df['Position'] > 0].head(5)
    pd.set_option('display.float_format', '{:.2%}'.format)
    print(longs[['Position', 'Volatility']])
    
    print("\n=== Top Short Positions ===")
    shorts = allocation_df[allocation_df['Position'] < 0].head(5)
    print(shorts[['Position', 'Volatility']])
    pd.reset_option('display.float_format')
    
    # Export to CSV
    allocation_df.to_csv('etf_allocation.csv')
    print("\nAllocation saved to etf_allocation.csv")
    
    return predictor, allocation_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Managed Futures ETF Dashboard')
    parser.add_argument('--build-models', action='store_true', 
                      help='Build new models instead of loading existing ones')
    args = parser.parse_args()
    
    dashboard(build_models=args.build_models)
    dashboard()