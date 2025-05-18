# portfolio_optimizer.py
"""
Enhanced portfolio optimization focusing on return generation 
while maintaining risk control for managed futures ETF strategy.
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_portfolio_risk(weights, returns, method='covariance'):
    """
    Calculate portfolio risk using covariance matrix or other methods.
    
    Parameters:
    -----------
    weights : np.array
        Asset weights
    returns : pd.DataFrame
        Historical returns
    method : str
        Risk calculation method ('covariance', 'var', or 'cvar')
        
    Returns:
    --------
    float
        Portfolio risk metric
    """
    if method == 'covariance':
        # Standard deviation using covariance matrix
        cov_matrix = returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance)
    
    elif method == 'var':
        # Value at Risk (VaR)
        portfolio_returns = np.dot(returns, weights)
        var_95 = np.percentile(portfolio_returns, 5)  # 95% VaR
        return -var_95  # Return positive value
    
    elif method == 'cvar':
        # Conditional Value at Risk (CVaR)
        portfolio_returns = np.dot(returns, weights)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        return -cvar_95  # Return positive value
    
    else:
        logger.warning(f"Unknown risk method '{method}'. Using covariance.")
        cov_matrix = returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance)

def predict_expected_returns(returns_data, method='historical', lookback=252):
    """
    Predict expected returns for assets.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        Historical returns data
    method : str
        Prediction method ('historical', 'ewma', or 'shrinkage')
    lookback : int
        Lookback period for historical returns
        
    Returns:
    --------
    pd.Series
        Expected returns for each asset
    """
    if method == 'historical':
        # Simple historical average
        return returns_data.iloc[-lookback:].mean()
    
    elif method == 'ewma':
        # Exponentially weighted moving average
        return returns_data.ewm(halflife=lookback/4).mean().iloc[-1]
    
    elif method == 'shrinkage':
        # Shrinkage estimator (blend of historical and prior)
        historical_mean = returns_data.iloc[-lookback:].mean()
        global_mean = historical_mean.mean()  # Cross-sectional mean
        
        # Calculate shrinkage intensity (simple version)
        shrinkage_intensity = 0.5
        
        # Blend historical and prior
        return historical_mean * (1 - shrinkage_intensity) + global_mean * shrinkage_intensity
    
    else:
        logger.warning(f"Unknown method '{method}'. Using historical.")
        return returns_data.iloc[-lookback:].mean()

def identify_strongly_trending_assets(returns_data, window=63, threshold=0.8):
    """
    Identify assets with strong trend characteristics.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        Historical returns data
    window : int
        Lookback window for trend calculation
    threshold : float
        Threshold for trend strength (0-1)
        
    Returns:
    --------
    list
        List of strongly trending assets
    """
    trending_assets = []
    
    for column in returns_data.columns:
        # Skip if we don't have enough data
        if len(returns_data[column].dropna()) < window:
            continue
        
        # Calculate cumulative return over window
        cumret = (1 + returns_data[column].iloc[-window:]).cumprod() - 1
        
        # Calculate trend strength (R-squared of linear fit)
        x = np.arange(len(cumret))
        y = cumret.values
        
        # Skip if we have NaN values
        if np.isnan(y).any():
            continue
        
        # Calculate linear regression
        try:
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared
            y_fit = slope * x + intercept
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_fit)**2)
            
            if ss_total == 0:  # Avoid division by zero
                r_squared = 0
            else:
                r_squared = 1 - (ss_residual / ss_total)
            
            # Check trend strength and direction
            if r_squared > threshold:
                trending_assets.append(column)
                
        except Exception as e:
            logger.warning(f"Error calculating trend for {column}: {str(e)}")
    
    return trending_assets

def market_in_strong_trend(market_data=None, market_index='SPY', window=63, threshold=0.7):
    """
    Determine if the overall market is in a strong trend.
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        Market index data
    market_index : str
        Market index to use
    window : int
        Lookback window
    threshold : float
        Threshold for trend strength
        
    Returns:
    --------
    bool
        True if market is in strong trend
    """
    # If no market data provided, assume no strong trend
    if market_data is None or market_index not in market_data.columns:
        return False
    
    # Get market returns
    market_returns = market_data[market_index].iloc[-window:]
    
    # Calculate trend strength
    x = np.arange(len(market_returns))
    y = (1 + market_returns).cumprod().values
    
    # Skip if we have NaN values
    if np.isnan(y).any():
        return False
    
    # Calculate linear regression
    try:
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared
        y_fit = slope * x + intercept
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_fit)**2)
        
        if ss_total == 0:  # Avoid division by zero
            r_squared = 0
        else:
            r_squared = 1 - (ss_residual / ss_total)
        
        # Check if market is in strong trend
        strong_trend = r_squared > threshold
        
        # Determine direction (positive or negative)
        trend_direction = 1 if slope > 0 else -1
        
        return strong_trend, trend_direction
        
    except Exception as e:
        logger.warning(f"Error calculating market trend: {str(e)}")
        return False, 0

def optimize_portfolio_with_return_target(allocations, returns_data, 
                                        risk_target=0.20, return_target=0.15,
                                        max_leverage=2.0, max_position=0.25,
                                        enable_trend_bias=True):
    """
    Optimize portfolio with explicit return target while respecting risk constraints.
    
    Parameters:
    -----------
    allocations : pd.Series or dict
        Initial allocation seeds (assets and signals)
    returns_data : pd.DataFrame
        Historical returns for assets
    risk_target : float
        Target portfolio volatility
    return_target : float
        Target portfolio return
    max_leverage : float
        Maximum allowed leverage
    max_position : float
        Maximum position size per asset
    enable_trend_bias : bool
        Whether to enable trend bias adjustment
        
    Returns:
    --------
    pd.Series
        Optimized portfolio allocations
    """
    # Convert allocations to Series if dict
    if isinstance(allocations, dict):
        allocations = pd.Series(allocations)
    
    # Ensure allocations and returns_data have compatible assets
    common_assets = [asset for asset in allocations.index if asset in returns_data.columns]
    
    if not common_assets:
        logger.error("No common assets between allocations and returns data")
        return allocations
    
    # Filter allocations and returns data to common assets
    allocations = allocations.loc[common_assets]
    filtered_returns = returns_data[common_assets]
    
    # Set minimum allocation to trending assets if requested
    trending_assets = []
    if enable_trend_bias:
        trending_assets = identify_strongly_trending_assets(filtered_returns)
        logger.info(f"Identified {len(trending_assets)} strongly trending assets: {trending_assets}")
    
    # Predict expected returns
    expected_returns = predict_expected_returns(filtered_returns)
    
    # Adjust direction according to signals from allocations
    signals = np.sign(allocations)
    adjusted_returns = expected_returns * signals
    
    # Handle missing or invalid values
    adjusted_returns = adjusted_returns.fillna(0)
    
    # Initial weights
    init_weights = allocations.abs() / allocations.abs().sum() * np.sign(allocations)
    
    # Optimization objective: maximize return while respecting risk target
    def objective(weights):
        # Calculate expected portfolio return (negative for minimization)
        portfolio_return = -np.sum(weights * adjusted_returns)
        return portfolio_return
    
    # Risk constraint: portfolio risk should be close to risk_target
    def risk_constraint(weights):
        portfolio_risk = calculate_portfolio_risk(weights, filtered_returns)
        return risk_target - portfolio_risk  # Risk should be at or below target
    
    # Return constraint: portfolio return should meet or exceed return_target
    def return_constraint(weights):
        portfolio_return = np.sum(weights * adjusted_returns)
        return portfolio_return - return_target  # Return should meet or exceed target
    
    # Leverage constraint: total leverage should not exceed max_leverage
    def leverage_constraint(weights):
        total_leverage = np.sum(np.abs(weights))
        return max_leverage - total_leverage  # Leverage should not exceed max
    
    # Create bounds for individual positions
    bounds = [(-max_position, max_position) for _ in range(len(common_assets))]
    
    # Create constraints
    constraints = [
        {'type': 'ineq', 'fun': risk_constraint},
        {'type': 'ineq', 'fun': return_constraint},
        {'type': 'ineq', 'fun': leverage_constraint}
    ]
    
    # Add trending asset constraints if applicable
    if trending_assets and enable_trend_bias:
        # For each trending asset, ensure minimum position size
        for asset in trending_assets:
            if asset in common_assets:
                idx = common_assets.index(asset)
                
                def min_trending_constraint(weights, idx=idx, sign=np.sign(allocations[asset])):
                    # Minimum absolute position of 0.05 in the right direction
                    return abs(weights[idx]) - 0.05 if np.sign(weights[idx]) == sign else -1
                
                constraints.append({'type': 'ineq', 'fun': min_trending_constraint})
    
    # Run optimization
    try:
        # Convert initial weights to numpy array
        init_weights_array = init_weights.values
        
        result = minimize(
            objective,
            init_weights_array,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            logger.info(f"Portfolio optimization successful: {result.message}")
            
            # Convert optimized weights back to Series
            optimized_weights = pd.Series(result.x, index=common_assets)
            
            # Check if return target is met
            expected_portfolio_return = np.sum(optimized_weights * adjusted_returns)
            expected_portfolio_risk = calculate_portfolio_risk(optimized_weights.values, filtered_returns)
            
            logger.info(f"Expected portfolio return: {expected_portfolio_return:.2%}")
            logger.info(f"Expected portfolio risk: {expected_portfolio_risk:.2%}")
            
            return optimized_weights
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return allocations
    
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {str(e)}")
        return allocations

def visualize_optimized_portfolio(original_allocations, optimized_allocations, returns_data=None):
    """
    Visualize comparison between original and optimized portfolio allocations.
    
    Parameters:
    -----------
    original_allocations : pd.Series
        Original portfolio allocations
    optimized_allocations : pd.Series
        Optimized portfolio allocations
    returns_data : pd.DataFrame, optional
        Historical returns data for risk-return metrics
        
    Returns:
    --------
    None
    """
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Sort allocations by optimized weight for clearer visualization
    common_assets = sorted(set(original_allocations.index).intersection(optimized_allocations.index),
                        key=lambda x: abs(optimized_allocations.get(x, 0)),
                        reverse=True)
    
    # Filter allocations
    orig_filtered = original_allocations.loc[common_assets]
    opt_filtered = optimized_allocations.loc[common_assets]
    
    # Plot original allocations
    colors1 = ['red' if x < 0 else 'green' for x in orig_filtered.values]
    ax1.barh(common_assets, orig_filtered.values, color=colors1)
    ax1.set_title('Original Portfolio Allocation')
    ax1.set_xlabel('Position Size')
    ax1.axvline(x=0, color='black', linestyle='--')
    
    # Add total leverage, long/short exposure, and net exposure
    orig_leverage = orig_filtered.abs().sum()
    orig_long = orig_filtered[orig_filtered > 0].sum()
    orig_short = orig_filtered[orig_filtered < 0].sum()
    orig_net = orig_long + orig_short
    
    ax1.text(
        0.02, 0.05,
        f"Total Leverage: {orig_leverage:.2f}x\n"
        f"Long Exposure: {orig_long:.2%}\n"
        f"Short Exposure: {orig_short:.2%}\n"
        f"Net Exposure: {orig_net:.2%}",
        transform=ax1.transAxes,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
    )
    
    # Plot optimized allocations
    colors2 = ['red' if x < 0 else 'green' for x in opt_filtered.values]
    ax2.barh(common_assets, opt_filtered.values, color=colors2)
    ax2.set_title('Optimized Portfolio Allocation')
    ax2.set_xlabel('Position Size')
    ax2.axvline(x=0, color='black', linestyle='--')
    
    # Add total leverage, long/short exposure, and net exposure
    opt_leverage = opt_filtered.abs().sum()
    opt_long = opt_filtered[opt_filtered > 0].sum()
    opt_short = opt_filtered[opt_filtered < 0].sum()
    opt_net = opt_long + opt_short
    
    ax2.text(
        0.02, 0.05,
        f"Total Leverage: {opt_leverage:.2f}x\n"
        f"Long Exposure: {opt_long:.2%}\n"
        f"Short Exposure: {opt_short:.2%}\n"
        f"Net Exposure: {opt_net:.2%}",
        transform=ax2.transAxes,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
    )
    
    # Add risk-return metrics if returns data is provided
    if returns_data is not None:
        common_returns = returns_data[common_assets]
        
        if not common_returns.empty:
            # Calculate metrics for original portfolio
            orig_portfolio_returns = common_returns.dot(orig_filtered)
            orig_annual_return = orig_portfolio_returns.mean() * 252
            orig_volatility = orig_portfolio_returns.std() * np.sqrt(252)
            orig_sharpe = orig_annual_return / orig_volatility if orig_volatility > 0 else 0
            
            # Calculate metrics for optimized portfolio
            opt_portfolio_returns = common_returns.dot(opt_filtered)
            opt_annual_return = opt_portfolio_returns.mean() * 252
            opt_volatility = opt_portfolio_returns.std() * np.sqrt(252)
            opt_sharpe = opt_annual_return / opt_volatility if opt_volatility > 0 else 0
            
            # Add summary of improvement
            fig.text(
                0.5, 0.01,
                f"Performance Comparison:\n"
                f"Original: Return={orig_annual_return:.2%}, Vol={orig_volatility:.2%}, Sharpe={orig_sharpe:.2f}\n"
                f"Optimized: Return={opt_annual_return:.2%}, Vol={opt_volatility:.2%}, Sharpe={opt_sharpe:.2f}\n"
                f"Improvement: Return Δ={opt_annual_return-orig_annual_return:.2%}, Sharpe Δ={opt_sharpe-orig_sharpe:.2f}",
                ha='center',
                bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round')
            )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()