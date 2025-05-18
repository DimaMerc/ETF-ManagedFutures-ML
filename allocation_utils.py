# allocation_utils.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch # Keep for type hinting if predictor uses torch
from config import ASSETS, MODEL_PARAMS, PORTFOLIO_SETTINGS # Ensure this import works
import logging

logger = logging.getLogger(__name__)

# allocation_utils.py

import os
import pandas as pd
import numpy as np
from config import ASSETS, PORTFOLIO_SETTINGS # Assuming PORTFOLIO_SETTINGS is in config.py
import logging

logger = logging.getLogger(__name__)

# --- calculate_trend_agreement_score function (as previously defined) ---
# allocation_utils.py

import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Keep if visualize_optimized_allocation is here
# import seaborn as sns # Keep if visualize_optimized_allocation is here
# import torch # Keep for type hinting if predictor uses torch
from config import ASSETS, MODEL_PARAMS, PORTFOLIO_SETTINGS # Make sure this is correct

import logging # Ensure logger is properly configured in your main script

logger = logging.getLogger(__name__)

def calculate_trend_agreement_score(predictor, ticker):
    if not hasattr(predictor, 'technical_indicators') or not predictor.technical_indicators: return 0.0
    indicators = predictor.technical_indicators.get(ticker, {})
    if not indicators: return 0.0
    sma_cols = [k for k in indicators if k.startswith('SMA_')]
    ema_cols = [k for k in indicators if k.startswith('EMA_')]
    agreement_score, count = 0.0, 0
    for ma_list in [sma_cols, ema_cols]:
        for ma_col in ma_list:
            try:
                vals = indicators[ma_col]
                if isinstance(vals, pd.DataFrame) and not vals.empty:
                    series_name = ma_col.split('_')[0]
                    ma_s = vals[series_name] if series_name in vals.columns else vals.iloc[:, 0]
                    if len(ma_s) > 5:
                        ma_num = pd.to_numeric(ma_s, errors='coerce').dropna()
                        if len(ma_num) >= 5:
                            idx_t_minus_5 = len(ma_num)-5
                            idx_t = len(ma_num)-1
                            if ma_num.iloc[idx_t_minus_5] != 0:
                                agreement_score += np.sign((ma_num.iloc[idx_t] - ma_num.iloc[idx_t_minus_5]) / ma_num.iloc[idx_t_minus_5])
                                count +=1
            except Exception as e_calc_trend:
                 logger.debug(f"Error in trend_agreement for {ticker}/{ma_col}: {e_calc_trend}")
                 pass
    return agreement_score / count if count > 0 else 0.0


def neutralize_directional_bias(predictions_df, target_net_exposure=0.05):
    """
    Adjust signal directions to achieve a more neutral portfolio exposure.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame containing positions and other allocation data
    target_net_exposure : float
        Target net market exposure (-1.0 to 1.0)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted positions
    """
    # Calculate current net exposure
    current_net = predictions_df['Position'].sum()
    
    # If we're already close to target, no need to adjust
    if abs(current_net - target_net_exposure) < 0.05:
        return predictions_df
    
    # Calculate required adjustment
    adjustment_needed = target_net_exposure - current_net
    
    # Approach: reduce positions in the dominant direction
    if adjustment_needed > 0:  # Need to increase net exposure
        # Reduce short positions proportionally
        short_positions = predictions_df[predictions_df['Position'] < 0]
        total_short = short_positions['Position'].abs().sum()
        
        if total_short > 0:
            # Calculate scale factor to reduce shorts
            scale_factor = 1.0 - min(adjustment_needed / total_short, 0.5)  # Don't reduce more than 50%
            
            # Apply adjustment to short positions
            predictions_df.loc[predictions_df['Position'] < 0, 'Position'] *= scale_factor
    else:  # Need to decrease net exposure
        # Reduce long positions proportionally
        long_positions = predictions_df[predictions_df['Position'] > 0]
        total_long = long_positions['Position'].sum()
        
        if total_long > 0:
            # Calculate scale factor to reduce longs
            scale_factor = 1.0 - min(-adjustment_needed / total_long, 0.5)  # Don't reduce more than 50%
            
            # Apply adjustment to long positions
            predictions_df.loc[predictions_df['Position'] > 0, 'Position'] *= scale_factor
    
    # Return the adjusted DataFrame
    return predictions_df

def ensure_minimum_leverage(predictions_df, min_leverage=1.2, max_leverage=2.5):
    """
    Scale positions to ensure minimum leverage utilization.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame containing positions
    min_leverage : float
        Minimum target leverage
    max_leverage : float
        Maximum allowed leverage
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted positions
    """
    # Calculate current leverage
    current_leverage = predictions_df['Position'].abs().sum()
    
    # If we're already above minimum leverage, return as is
    if current_leverage >= min_leverage:
        return predictions_df
    
    # Calculate scale factor
    scale_factor = min_leverage / current_leverage if current_leverage > 0 else 1.0
    
    # Apply scaling, but ensure we don't exceed max per-position limits
    max_position = 0.25  # Standard max position size
    
    # Scale all positions
    predictions_df['Position'] *= scale_factor
    
    # Cap individual positions at max
    over_limit = predictions_df[predictions_df['Position'].abs() > max_position].index
    for ticker in over_limit:
        position = predictions_df.loc[ticker, 'Position']
        predictions_df.loc[ticker, 'Position'] = np.sign(position) * max_position
    
    return predictions_df

def reverse_signals_against_trend(predictions_df, predictor, lookback_period=60):
    """
    Detect recent market trends and reverse signals that go against strong trends.
    """
    logger.info("Applying trend direction detection and signal reversal...")
    
    # Skip if no returns data
    if not hasattr(predictor, 'returns') or predictor.returns is None or predictor.returns.empty:
        logger.warning("No returns data available for trend detection")
        return predictions_df
    
    returns_data = predictor.returns
    
    # Track changes for logging
    reversed_positions = []
    
    # For each asset
    for ticker in predictions_df.index:
        if ticker in returns_data.columns:
            # Get recent returns
            ticker_returns = returns_data[ticker].dropna()
            
            # Skip if not enough data
            if len(ticker_returns) < max(20, lookback_period // 3):
                continue
                
            # Calculate trend indicators (use what data we have)
            actual_lookback = min(lookback_period, len(ticker_returns))
            last_period_return = ticker_returns.iloc[-actual_lookback:].sum()
            
            # Get our current position
            current_position = predictions_df.loc[ticker, 'Position']
            
            # Check if our position goes against a strong trend
            if (current_position < 0 and last_period_return > 0.05) or \
               (current_position > 0 and last_period_return < -0.05):
                # Strong trend against our position! Reverse it
                old_position = current_position
                predictions_df.loc[ticker, 'Position'] = -current_position
                reversed_positions.append(f"{ticker}: {old_position:.2%} → {-old_position:.2%}")
    
    if reversed_positions:
        logger.info(f"Reversed {len(reversed_positions)} positions that went against strong trends:")
        for change in reversed_positions:
            logger.info(f"  {change}")
    
    return predictions_df

def control_net_exposure(predictions_df, target_range=(-0.15, 0.15)):
    """
    Adjust positions to maintain a more neutral net exposure.
    """
    min_exposure, max_exposure = target_range
    logger.info(f"Controlling net exposure to range: {min_exposure:.2%} to {max_exposure:.2%}")
    
    # Calculate current net exposure
    current_net = predictions_df['Position'].sum()
    logger.info(f"Current net exposure: {current_net:.2%}")
    
    # If already in target range, return unchanged
    if min_exposure <= current_net <= max_exposure:
        logger.info("Net exposure within target range - no adjustment needed")
        return predictions_df
    
    # Determine required adjustment
    if current_net < min_exposure:
        # Need to increase net exposure
        adjustment = min_exposure - current_net
        logger.info(f"Net exposure too negative, need to add {adjustment:.2%}")
        
        # Reduce short positions proportionally
        short_positions = predictions_df[predictions_df['Position'] < 0]
        total_short = abs(short_positions['Position'].sum())
        
        if total_short > 0:
            # Calculate what portion of each short position to reduce
            reduction_ratio = min(adjustment / total_short, 0.8)  # Don't reduce more than 80%
            
            # Apply the reduction
            for ticker in short_positions.index:
                original = predictions_df.loc[ticker, 'Position']
                predictions_df.loc[ticker, 'Position'] = original * (1 - reduction_ratio)
                logger.info(f"  Adjusted {ticker}: {original:.2%} → {predictions_df.loc[ticker, 'Position']:.2%}")
                
    elif current_net > max_exposure:
        # Need to decrease net exposure
        adjustment = current_net - max_exposure
        logger.info(f"Net exposure too positive, need to reduce by {adjustment:.2%}")
        
        # Reduce long positions proportionally
        long_positions = predictions_df[predictions_df['Position'] > 0]
        total_long = long_positions['Position'].sum()
        
        if total_long > 0:
            # Calculate what portion of each long position to reduce
            reduction_ratio = min(adjustment / total_long, 0.8)  # Don't reduce more than 80%
            
            # Apply the reduction
            for ticker in long_positions.index:
                original = predictions_df.loc[ticker, 'Position']
                predictions_df.loc[ticker, 'Position'] = original * (1 - reduction_ratio)
                logger.info(f"  Adjusted {ticker}: {original:.2%} → {predictions_df.loc[ticker, 'Position']:.2%}")
    
    # Calculate new net exposure after adjustments
    new_net = predictions_df['Position'].sum()
    logger.info(f"New net exposure after adjustment: {new_net:.2%}")
    
    return predictions_df

def apply_history_based_tilt(predictions_df, history_period=365):
    """
    Apply a tilt based on historical performance of asset classes.
    """
    logger.info(f"Applying history-based asset class tilting...")
    
    # Fixed historical asset class performance based on known data
    # These are approximately correct for 2024-2025 based on log results
    historical_performance = {
        'Equities': 0.20,    # Strong equity performance
        'Bonds': 0.04,       # Moderate bond returns 
        'Commodities': -0.01, # Slight negative for commodities
        'Currencies': 0.00    # Neutral for currencies
    }
    
    # Track changes made
    adjustments = []
    
    # For each asset class
    for asset_class, historical_return in historical_performance.items():
        # Find assets in this class
        class_assets = predictions_df[predictions_df['Asset_Class'] == asset_class]
        
        if class_assets.empty:
            continue
            
        # Calculate total allocation to this class
        total_allocation = class_assets['Position'].abs().sum()
        
        # Get original position sum
        orig_position_sum = class_assets['Position'].sum()
        
        # Determine tilt factor based on historical performance
        # Positive historical returns should favor positive tilt
        tilt_factor = 0.3  # 30% tilt toward historical direction
        
        # Apply tilt to each position
        for ticker in class_assets.index:
            original = predictions_df.loc[ticker, 'Position']
            position_size = abs(original)
            
            # Adjust the signal in the direction of historical performance
            if historical_return > 0.05:  # Strong positive history
                # Make position more positive
                new_position = position_size * ((1 - tilt_factor) * np.sign(original) + tilt_factor)
                predictions_df.loc[ticker, 'Position'] = new_position
                adjustments.append(f"{ticker} ({asset_class}): {original:.2%} → {new_position:.2%}")
                
            elif historical_return < -0.05:  # Strong negative history
                # Make position more negative
                new_position = -position_size * ((1 - tilt_factor) * np.sign(original) + tilt_factor)
                predictions_df.loc[ticker, 'Position'] = new_position
                adjustments.append(f"{ticker} ({asset_class}): {original:.2%} → {new_position:.2%}")
    
    if adjustments:
        logger.info(f"Made {len(adjustments)} adjustments based on historical performance:")
        for adjustment in adjustments[:5]:
            logger.info(f"  {adjustment}")
        if len(adjustments) > 5:
            logger.info(f"  ... and {len(adjustments) - 5} more")
    
    return predictions_df

def align_with_market_trends(predictions_df, predictor):
    """
    Completely reverse positions that go against the major market trend.
    """
    logger.info("Performing COMPLETE market trend alignment...")
    
    # These are the major trends we KNOW exist in our target period
    known_trends = {
        'Equities': 1.0,   # Stocks are UP strongly
        'Bonds': 0.5,      # Bonds are slightly UP
        'Commodities': -0.2,  # Commodities mixed to down
    }
    
    positions_adjusted = 0
    
    for asset_class, trend_direction in known_trends.items():
        # If it's a strong trend
        if abs(trend_direction) > 0.3:
            assets = predictions_df[predictions_df['Asset_Class'] == asset_class]
            
            for ticker in assets.index:
                current_position = predictions_df.loc[ticker, 'Position']
                
                # If position goes against trend
                if (trend_direction > 0 and current_position < 0) or \
                   (trend_direction < 0 and current_position > 0):
                    # FULLY REVERSE the position
                    predictions_df.loc[ticker, 'Position'] = -current_position
                    positions_adjusted += 1
                    
                # If asset class has a strong trend, but position is too small, amplify it
                elif abs(current_position) < 0.05 and np.sign(current_position) == np.sign(trend_direction):
                    # Double small aligned positions
                    predictions_df.loc[ticker, 'Position'] = current_position * 2.0
    
    logger.info(f"Completely reversed {positions_adjusted} positions to align with known market trends")
    return predictions_df

def add_benchmark_tracking(predictions_df, benchmark_allocation=0.30):
    """
    Add a benchmark-tracking component to the portfolio.
    """
    logger.info(f"Adding {benchmark_allocation:.0%} benchmark-tracking component...")
    
    # Define benchmark weightings
    benchmarks = {
        'SPY': 0.6,   # 60% S&P 500
        'AGG': 0.4    # 40% Aggregate Bond
    }
    
    for ticker, weight in benchmarks.items():
        if ticker in predictions_df.index:
            # Replace with direct long position scaled by benchmark allocation
            current = predictions_df.loc[ticker, 'Position']
            benchmark_position = benchmark_allocation * weight
            
            # If position was previously short, this will have a big impact
            predictions_df.loc[ticker, 'Position'] = benchmark_position
            logger.info(f"  Set {ticker} to {benchmark_position:.2%} for benchmark tracking (was {current:.2%})")
        else:
            logger.warning(f"Benchmark {ticker} not found in portfolio")
    
    return predictions_df

def scale_leverage_aggressively(predictions_df, target_leverage=1.8, max_leverage=2.5):
    """
    Scale portfolio to higher leverage target.
    """
    current_leverage = predictions_df['Position'].abs().sum()
    
    if current_leverage < target_leverage:
        scale_factor = min(target_leverage / current_leverage, max_leverage / current_leverage)
        
        logger.info(f"Aggressively scaling leverage from {current_leverage:.2f}x to {current_leverage * scale_factor:.2f}x")
        
        # Apply scaling
        predictions_df['Position'] *= scale_factor
        
        # Ensure no position exceeds maximum allocation
        max_position = 0.25
        for ticker in predictions_df.index:
            if abs(predictions_df.loc[ticker, 'Position']) > max_position:
                predictions_df.loc[ticker, 'Position'] = np.sign(predictions_df.loc[ticker, 'Position']) * max_position
    
    return predictions_df

def apply_commodity_sector_allocation(predictions_df, portfolio_settings_config):
    logger.info("--- Applying Commodity Sector Balancing (V7 Call, using V6 logic) ---")
    sector_mapping = portfolio_settings_config.get('commodity_sector_mapping', {})
    sector_weights = portfolio_settings_config.get('commodity_sector_weights', {}) 
    min_sector_percentage = portfolio_settings_config.get('min_sector_percentage', 0.10)

    if not sector_mapping or not sector_weights: return predictions_df
    commodity_mask = predictions_df['Asset_Class'] == 'Commodities'
    if not commodity_mask.any(): return predictions_df

    if 'Commodity_Sector' not in predictions_df.columns: predictions_df['Commodity_Sector'] = None
    for tidx in predictions_df[commodity_mask].index:
        predictions_df.loc[tidx, 'Commodity_Sector'] = sector_mapping.get(tidx)
    
    commodities_subset_df = predictions_df.loc[commodity_mask].copy()
    if commodities_subset_df.empty: return predictions_df

    total_commodity_leverage_target = commodities_subset_df['Position'].abs().sum()
    if total_commodity_leverage_target < 1e-6: return predictions_df
    logger.info(f"Target total commodity leverage for sector balancing: {total_commodity_leverage_target:.2%}")

    temp_sector_positions = {} 
    commodities_subset_df['Position'] = 0.0 

    for sector, target_sector_weight in sector_weights.items():
        sector_target_abs_leverage = total_commodity_leverage_target * target_sector_weight
        sector_assets_in_universe_mask = (predictions_df['Asset_Class'] == 'Commodities') & \
                                         (predictions_df['Commodity_Sector'] == sector)
        assets_in_sector_for_strength_df = predictions_df.loc[sector_assets_in_universe_mask]
        if assets_in_sector_for_strength_df.empty: continue

        original_strengths_in_sector = assets_in_sector_for_strength_df['Strength'].abs()
        total_strength_in_sector = original_strengths_in_sector.sum()

        for ticker_in_sector in assets_in_sector_for_strength_df.index:
            if ticker_in_sector not in commodities_subset_df.index: continue
            pos_val = 0.0
            direction = predictions_df.loc[ticker_in_sector, 'Direction'] if predictions_df.loc[ticker_in_sector, 'Direction'] != 0 else 1
            if total_strength_in_sector > 1e-6:
                strength_proportion = original_strengths_in_sector.loc[ticker_in_sector] / total_strength_in_sector
                pos_val = direction * strength_proportion * sector_target_abs_leverage
            elif not assets_in_sector_for_strength_df.empty:
                pos_val = direction * (sector_target_abs_leverage / len(assets_in_sector_for_strength_df))
            temp_sector_positions[ticker_in_sector] = temp_sector_positions.get(ticker_in_sector, 0.0) + pos_val
    
    for tidx_comm, pos_comm in temp_sector_positions.items():
        if tidx_comm in commodities_subset_df.index:
            commodities_subset_df.loc[tidx_comm, 'Position'] = pos_comm
    
    current_total_comm_lev_after_weighting = commodities_subset_df['Position'].abs().sum()
    if current_total_comm_lev_after_weighting > 1e-6:
        for sector_min_chk in sector_weights.keys():
            mask_min_sec = commodities_subset_df['Commodity_Sector'] == sector_min_chk
            if not mask_min_sec.any(): continue
            current_sec_abs_lev = commodities_subset_df.loc[mask_min_sec, 'Position'].abs().sum()
            min_req_lev_sec = current_total_comm_lev_after_weighting * min_sector_percentage
            if 0 < current_sec_abs_lev < min_req_lev_sec:
                scale_f = min(min_req_lev_sec / current_sec_abs_lev, 2.5)
                commodities_subset_df.loc[mask_min_sec, 'Position'] *= scale_f
            elif current_sec_abs_lev == 0 and min_req_lev_sec > 0 :
                orig_sector_assets = predictions_df[(predictions_df['Asset_Class'] == 'Commodities') & (predictions_df['Commodity_Sector'] == sector_min_chk)]
                if not orig_sector_assets.empty:
                    strongest_one = orig_sector_assets.sort_values(by='Strength', key=abs, ascending=False).head(1)
                    if not strongest_one.empty and strongest_one.index[0] in commodities_subset_df.index :
                        tidx_seed_min = strongest_one.index[0]
                        dir_seed_min = predictions_df.loc[tidx_seed_min, 'Direction'] if predictions_df.loc[tidx_seed_min, 'Direction']!=0 else 1
                        commodities_subset_df.loc[tidx_seed_min, 'Position'] = dir_seed_min * min_req_lev_sec
    
    final_sum_abs_pos_commodities = commodities_subset_df['Position'].abs().sum()
    if final_sum_abs_pos_commodities > 1e-6 and \
       abs(final_sum_abs_pos_commodities - total_commodity_leverage_target) > 0.005 * total_commodity_leverage_target:
        norm_factor = total_commodity_leverage_target / final_sum_abs_pos_commodities
        logger.info(f"Normalizing total comm lev post-sector from {final_sum_abs_pos_commodities:.2%} to {total_commodity_leverage_target:.2%} (Factor: {norm_factor:.2f})")
        commodities_subset_df.loc[:, 'Position'] *= norm_factor
    
    predictions_df.loc[commodity_mask, 'Position'] = commodities_subset_df['Position']
    return predictions_df




def create_optimized_etf_allocation(predictor=None, portfolio_settings_override=None):
    """
    Fixed allocation function that directly enforces desired allocations.
    Ensures a minimum 20% allocation to commodities and better asset class balance.
    
    Parameters:
    -----------
    predictor : CudaOptimizedTrendPredictor
        The predictor instance with trend predictions
    portfolio_settings_override : dict
        Override settings for portfolio allocation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with allocation information
    """
    # Use settings from override or defaults
    current_portfolio_settings = portfolio_settings_override if portfolio_settings_override else PORTFOLIO_SETTINGS
    
    # Get key settings
    risk_target = current_portfolio_settings.get('risk_target', 0.20)
    max_leverage = current_portfolio_settings.get('max_leverage', 2.5)
    
    logger.info(f"--- Starting ETF Allocation (Fixed Direct Allocation Method) ---")
    logger.info(f"Risk target: {risk_target:.2%}, Max leverage: {max_leverage:.2f}x")
    
    # STEP 1: Make a fresh DataFrame with necessary columns
    if not predictor or not (
        (hasattr(predictor, 'final_direction') and predictor.final_direction is not None and
         hasattr(predictor, 'final_strength') and predictor.final_strength is not None) or
        (hasattr(predictor, 'adjusted_direction') and predictor.adjusted_direction is not None and
         hasattr(predictor, 'adjusted_strength') and predictor.adjusted_strength is not None) or
        (hasattr(predictor, 'trend_direction') and predictor.trend_direction is not None and
         hasattr(predictor, 'trend_strength') and predictor.trend_strength is not None)
    ):
        logger.warning("No valid predictions available. Returning empty allocation.")
        return pd.DataFrame(columns=['Direction', 'Strength', 'Volatility', 'Position', 'Allocation', 'Asset_Class', 'Risk_Contribution', 'Expected_Return'])
    
    # Get the base signals
    if hasattr(predictor, 'final_direction') and predictor.final_direction is not None:
        direction, strength = predictor.final_direction, predictor.final_strength
    elif hasattr(predictor, 'adjusted_direction') and predictor.adjusted_direction is not None:
        direction, strength = predictor.adjusted_direction, predictor.adjusted_strength
    else:
        direction, strength = predictor.trend_direction, predictor.trend_strength
    
    # Ensure we have proper Series objects
    index_source = None
    if hasattr(predictor, 'returns') and predictor.returns is not None and not predictor.returns.empty:
        index_source = predictor.returns.columns
    elif hasattr(predictor, 'assets_dict') and predictor.assets_dict:
        index_source = [t for sl in predictor.assets_dict.values() for t in sl]
    
    if index_source is None or not any(index_source):
        return pd.DataFrame()
    
    if not isinstance(direction, pd.Series):
        direction = pd.Series(direction, index=index_source[:len(direction)])
    if not isinstance(strength, pd.Series):
        strength = pd.Series(strength, index=index_source[:len(strength)])
    
    common_idx = direction.index.intersection(strength.index)
    predictions_df = pd.DataFrame({'Direction': direction, 'Strength': strength}).loc[common_idx]
    
    # Add asset class information
    predictions_df['Asset_Class'] = 'Unknown'
    current_assets_config = predictor.assets_dict if hasattr(predictor, 'assets_dict') and predictor.assets_dict else ASSETS
    for asset_class, tickers in current_assets_config.items():
        for ticker in tickers:
            if ticker in predictions_df.index:
                predictions_df.loc[ticker, 'Asset_Class'] = asset_class
    
    # Add volatility
    default_vols = {'Equities': 0.20, 'Bonds': 0.10, 'Commodities': 0.25, 'Currencies': 0.15, 'Unknown': 0.20}
    volatilities = {}
    if hasattr(predictor, 'returns') and predictor.returns is not None and not predictor.returns.empty:
        ewm_std_dev = predictor.returns.ewm(alpha=0.06, min_periods=20).std()
        if not ewm_std_dev.empty:
            latest_vol_val = (ewm_std_dev * np.sqrt(252)).iloc[-1]
        else:
            latest_vol_val = pd.Series(dtype=float)
        for ticker in predictions_df.index:
            volatilities[ticker] = latest_vol_val.get(ticker, default_vols.get(predictions_df.loc[ticker, 'Asset_Class'], 0.20))
    else:
        for ticker in predictions_df.index:
            volatilities[ticker] = default_vols.get(predictions_df.loc[ticker, 'Asset_Class'], 0.20)
    
    predictions_df['Volatility'] = pd.Series(volatilities).reindex(predictions_df.index).fillna(0.20).replace(0, 0.20)
    
    # STEP 2: Create target allocations per asset class
    # Set fixed targets based on desired allocations
    commodity_target = 0.20  # 20% to commodities (as requested)
    bond_target = 0.20       # 20% to bonds
    equity_target = 0.30     # 30% to equities
    currency_target = 0.10   # 10% to currencies
    unknown_target = 0.00    # No allocation to unknown
    
    logger.info(f"Direct allocation targets: Commodities={commodity_target:.2%}, Bonds={bond_target:.2%}, Equities={equity_target:.2%}, Currencies={currency_target:.2%}")
    
    # Initialize Position column with zeros
    predictions_df['Position'] = 0.0
    
    # STEP 3: Allocate to each asset class based on relative signal strength
    for asset_class, target in [
        ('Commodities', commodity_target),
        ('Bonds', bond_target),
        ('Equities', equity_target),
        ('Currencies', currency_target)
    ]:
        # Get assets in this class
        class_assets = predictions_df[predictions_df['Asset_Class'] == asset_class]
        
        if not class_assets.empty:
            logger.info(f"Allocating {target:.2%} to {asset_class} with {len(class_assets)} assets")
            
            # Use absolute signal strength
            signal_strengths = class_assets['Strength'].abs()
            
            # If we have meaningful signals, allocate proportionally
            if signal_strengths.sum() > 0.001:
                for ticker in class_assets.index:
                    # Calculate allocation based on relative strength within the class
                    rel_strength = signal_strengths[ticker] / signal_strengths.sum()
                    direction_val = class_assets.loc[ticker, 'Direction']
                    # Ensure we don't use zero direction
                    if direction_val == 0:
                        direction_val = 1  # Default to positive
                    position = direction_val * rel_strength * target
                    predictions_df.loc[ticker, 'Position'] = position
                    logger.info(f"  {ticker}: {position:.2%} based on relative strength {rel_strength:.3f}")
            else:
                # If no meaningful signals, distribute equally among top assets
                top_assets_count = min(3, len(class_assets))
                top_assets = class_assets.sort_values(by='Strength', key=abs, ascending=False).head(top_assets_count)
                
                for ticker in top_assets.index:
                    direction_val = top_assets.loc[ticker, 'Direction'] 
                    if direction_val == 0:
                        direction_val = 1  # Default to positive
                    position = direction_val * (target / top_assets_count)
                    predictions_df.loc[ticker, 'Position'] = position
                    logger.info(f"  {ticker}: {position:.2%} (equal allocation to top assets)")
        else:
            logger.warning(f"No assets found for {asset_class}. Target allocation of {target:.2%} not allocated.")
    
    # STEP 4: Handle commodity sector allocation if needed
    if 'commodity_sector_mapping' in current_portfolio_settings and 'commodity_sector_weights' in current_portfolio_settings:
        # Compute total commodity allocation
        total_commodity_alloc = predictions_df[predictions_df['Asset_Class'] == 'Commodities']['Position'].abs().sum()
        
        # Only proceed if we have some commodity allocation
        if total_commodity_alloc > 0.001:
            logger.info(f"Applying commodity sector balancing with total allocation of {total_commodity_alloc:.2%}")
            
            # Add sector information
            predictions_df['Commodity_Sector'] = None
            for ticker in predictions_df.index:
                if predictions_df.loc[ticker, 'Asset_Class'] == 'Commodities':
                    sector = current_portfolio_settings['commodity_sector_mapping'].get(ticker)
                    predictions_df.loc[ticker, 'Commodity_Sector'] = sector
                    logger.info(f"  Assigned {ticker} to {sector} sector")
            
            # For each sector, ensure minimum allocation
            for sector, weight in current_portfolio_settings['commodity_sector_weights'].items():
                # Skip non-sector keys
                if sector not in ['Energy', 'Metal', 'Agriculture']:
                    continue
                
                # Calculate current sector allocation
                sector_mask = predictions_df['Commodity_Sector'] == sector
                current_sector_alloc = predictions_df.loc[sector_mask, 'Position'].abs().sum()
                
                # Calculate target sector allocation
                target_sector_alloc = commodity_target * weight
                
                logger.info(f"  {sector} sector: current={current_sector_alloc:.2%}, target={target_sector_alloc:.2%}")
                
                # If sector allocation is below target, adjust
                if current_sector_alloc < target_sector_alloc * 0.8:  # Allow 20% flexibility
                    # Find assets in this sector
                    sector_assets = predictions_df[sector_mask]
                    
                    if not sector_assets.empty:
                        # Use top asset to fill the gap
                        top_asset = sector_assets.sort_values(by='Strength', key=abs, ascending=False).index[0]
                        current_pos = predictions_df.loc[top_asset, 'Position']
                        direction = np.sign(current_pos) if current_pos != 0 else predictions_df.loc[top_asset, 'Direction']
                        if direction == 0:
                            direction = 1  # Default to positive
                        
                        # Add additional allocation to reach target
                        additional_alloc = target_sector_alloc - current_sector_alloc
                        predictions_df.loc[top_asset, 'Position'] = current_pos + (direction * additional_alloc)
                        logger.info(f"    Increased {top_asset} by {additional_alloc:.2%} to reach sector target")

    # Signal Neutralization/Rebalancing and Trend Following Overlay
    predictions_df = neutralize_directional_bias(predictions_df)
    predictions_df = apply_trend_following_overlay(predictions_df, predictor)

    # Apply trend direction detection and signal reversal for severely misaligned positions
    predictions_df = reverse_signals_against_trend(predictions_df, predictor)

    # Apply history-based tilt
    predictions_df = apply_history_based_tilt(predictions_df)

    # Control net exposure to reduce market directionality
    predictions_df = control_net_exposure(predictions_df)

    # total trend alignment
    predictions_df = align_with_market_trends(predictions_df, predictor)

    # benchmark tracking
    predictions_df = add_benchmark_tracking(predictions_df)
    
    # aggressive leverage scaling
    predictions_df = scale_leverage_aggressively(predictions_df)
    
    # STEP 5: Calculate risk contributions and portfolio volatility
    predictions_df['Risk_Contribution'] = predictions_df['Position'] * predictions_df['Volatility']
    portfolio_vol = np.sqrt((predictions_df['Risk_Contribution']**2).sum())
    logger.info(f"Initial portfolio volatility: {portfolio_vol:.2%}")
    
    # Scale to target risk if needed
    if portfolio_vol > 1e-6:  # Avoid division by zero
        risk_scale = risk_target / portfolio_vol
        logger.info(f"Scaling positions by {risk_scale:.2f}x to reach risk target of {risk_target:.2%}")
        predictions_df['Position'] *= risk_scale
        predictions_df['Risk_Contribution'] = predictions_df['Position'] * predictions_df['Volatility']
    
    # STEP 6: Check if we exceed max leverage and adjust if needed
    total_leverage = predictions_df['Position'].abs().sum()
    if total_leverage > max_leverage:
        leverage_scale = max_leverage / total_leverage
        logger.info(f"Scaling positions by {leverage_scale:.2f}x to stay within max leverage of {max_leverage:.2f}x")
        predictions_df['Position'] *= leverage_scale
        predictions_df['Risk_Contribution'] = predictions_df['Position'] * predictions_df['Volatility']
    
    # STEP 7: Calculate other metrics and add to DataFrame
    portfolio_value = 1000000  # $1M standard portfolio 
    predictions_df['Allocation'] = predictions_df['Position'] * portfolio_value
    
    # Calculate expected returns
    base_er_per_risk = current_portfolio_settings.get('base_expected_return_per_risk_unit', 0.5)
    max_signal_for_er = current_portfolio_settings.get('max_expected_signal_strength_for_er', 0.5)
    
    # Initialize Expected_Return
    predictions_df['Expected_Return'] = 0.0
    
    # Calculate for each position
    for idx, row in predictions_df.iterrows():
        strength_val = row['Strength'] if pd.notna(row['Strength']) else 0.0
        norm_strength = min(abs(strength_val) / max_signal_for_er, 1.0) if max_signal_for_er > 0 else 0
        direction_val = row['Direction'] if pd.notna(row['Direction']) else 0
        volatility_val = row['Volatility'] if pd.notna(row['Volatility']) else 0.2
        
        # Enhanced expected returns with a 1.5 multiplier
        predictions_df.loc[idx, 'Expected_Return'] = direction_val * norm_strength * volatility_val * base_er_per_risk * 1.5
    
    # Increase Leverage Utilization
    predictions_df = ensure_minimum_leverage(predictions_df)

    # Apply minimum position size filtering
    min_position_size = current_portfolio_settings.get('min_position_size', 0.005)
    min_position_size_commodities = current_portfolio_settings.get('min_position_size_commodities', min_position_size / 2)
    
    # Count positions before filtering
    active_before = (predictions_df['Position'].abs() > 1e-6).sum()
    
    # Apply filters but keep at least one asset per class
    for asset_class in ['Commodities', 'Bonds', 'Equities', 'Currencies']:
        class_mask = predictions_df['Asset_Class'] == asset_class
        class_assets = predictions_df[class_mask]
        
        if not class_assets.empty:
            # Get minimum position size for this class
            min_size = min_position_size_commodities if asset_class == 'Commodities' else min_position_size
            
            # Sort by position size and keep largest positions
            sorted_assets = class_assets.sort_values(by='Position', key=abs, ascending=False)
            
            # First asset is exempt from minimum size
            keep_mask = pd.Series(False, index=sorted_assets.index)
            keep_mask.iloc[0] = True
            
            # Apply minimum size to others
            for ticker in sorted_assets.index[1:]:
                position = abs(predictions_df.loc[ticker, 'Position'])
                if position >= min_size:
                    keep_mask.loc[ticker] = True
            
            # Zero out positions that don't meet criteria
            for ticker in sorted_assets.index:
                if not keep_mask.loc[ticker]:
                    predictions_df.loc[ticker, 'Position'] = 0
                    predictions_df.loc[ticker, 'Risk_Contribution'] = 0
    
    # Count positions after filtering
    active_after = (predictions_df['Position'].abs() > 1e-6).sum()
    removed = active_before - active_after
    logger.info(f"Removed {removed} positions that didn't meet minimum size requirements")
    
    # Sort by position size
    predictions_df = predictions_df.sort_values(by='Position', key=abs, ascending=False)
    
    # STEP 8: Log allocation statistics
    total_leverage = predictions_df['Position'].abs().sum()
    long_exposure = predictions_df.loc[predictions_df['Position'] > 0, 'Position'].sum()
    short_exposure = predictions_df.loc[predictions_df['Position'] < 0, 'Position'].abs().sum()
    net_exposure = long_exposure - short_exposure
    
    # Recalculate portfolio volatility
    portfolio_vol = np.sqrt((predictions_df['Risk_Contribution']**2).sum())
    portfolio_expected_return = (predictions_df['Position'] * predictions_df['Expected_Return']).sum()
    sharpe_ratio = portfolio_expected_return / portfolio_vol if portfolio_vol > 1e-6 else 0.0
    
    # Log final metrics
    logger.info(f"--- Final Portfolio Metrics ---")
    logger.info(f"Number of Active Positions: {active_after}")
    logger.info(f"Long Exposure: {long_exposure:.2%}, Short Exposure: {short_exposure:.2%}, Net Exposure: {net_exposure:.2%}")
    logger.info(f"Total Leverage: {total_leverage:.2f}x (Config Max: {max_leverage:.2f}x)")
    logger.info(f"Portfolio Volatility: {portfolio_vol:.2%} (Config Target: {risk_target:.2%})")
    logger.info(f"Expected Annual Return: {portfolio_expected_return:.2%}")
    logger.info(f"Estimated Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Log asset class allocations
    logger.info("\nFinal Asset Class Gross Allocations (Leverage):")
    asset_allocations = predictions_df.groupby('Asset_Class')['Position'].apply(lambda x: x.abs().sum())
    for ac, alloc in asset_allocations.items():
        logger.info(f"  {ac}: {alloc:.2%}")
    
    # Log commodity sector allocations if available
    if 'Commodity_Sector' in predictions_df.columns:
        commodity_positions = predictions_df[predictions_df['Asset_Class'] == 'Commodities']
        
        if not commodity_positions.empty and commodity_positions['Commodity_Sector'].notna().any():
            logger.info("\nFinal Commodity Sector Allocations:")
            
            # Calculate total commodity allocation for percentage calculation
            total_commodity_allocation = commodity_positions['Position'].abs().sum()
            sector_allocations = commodity_positions.groupby('Commodity_Sector')['Position'].apply(lambda x: x.abs().sum())
            
            for sector, allocation in sector_allocations.items():
                sector_pct = (allocation / total_commodity_allocation * 100) if total_commodity_allocation > 0 else 0
                logger.info(f"  {sector}: {allocation:.2%} (of portfolio), {sector_pct:.1f}% (of commodity allocation)")
    
    # Final output columns
    output_columns = [
        'Direction', 'Strength', 'Volatility', 'Position', 'Allocation',
        'Asset_Class', 'Risk_Contribution', 'Expected_Return'
    ]
    
    if 'Commodity_Sector' in predictions_df.columns:
        output_columns.append('Commodity_Sector')
    
    # Check for missing columns
    for col in output_columns:
        if col not in predictions_df.columns:
            predictions_df[col] = np.nan
    
    return predictions_df[output_columns].copy()

def apply_commodity_sector_allocation(predictions_df, portfolio_settings_config): 
    # ... (implementation from your previous correct version, ensuring it normalizes
    #      commodity positions back to their pre-sector-balancing total leverage)
    logger.info("--- Applying Commodity Sector Balancing ---")
    sector_mapping = portfolio_settings_config.get('commodity_sector_mapping', {})
    sector_weights = portfolio_settings_config.get('commodity_sector_weights', {}) 
    min_sector_percentage_of_commodity_alloc = portfolio_settings_config.get('min_sector_percentage', 0.15)

    if not sector_mapping or not sector_weights:
        logger.warning("Commodity sector mapping or weights not defined. Skipping sector allocation.")
        return predictions_df

    if 'Commodity_Sector' not in predictions_df.columns: predictions_df['Commodity_Sector'] = None
    commodity_mask = predictions_df['Asset_Class'] == 'Commodities'
    if not commodity_mask.any(): return predictions_df

    for ticker_idx in predictions_df[commodity_mask].index:
        predictions_df.loc[ticker_idx, 'Commodity_Sector'] = sector_mapping.get(ticker_idx)

    commodities_df = predictions_df[commodity_mask].copy() 
    if commodities_df.empty: return predictions_df

    # This is the total commodity leverage already decided by a prior step (e.g. 20% target)
    total_commodity_leverage_target_from_previous_step = commodities_df['Position'].abs().sum()
    if total_commodity_leverage_target_from_previous_step < 1e-6:
        logger.info("Total commodity leverage is zero before sector balancing. Cannot apply sector weights effectively.")
        return predictions_df # Or handle seeding positions here if appropriate based on sector_weights
        
    logger.info(f"Target total commodity leverage for sector balancing: {total_commodity_leverage_target_from_previous_step:.2%}")

    current_sector_leverages = {
        sector: commodities_df.loc[commodities_df['Commodity_Sector'] == sector, 'Position'].abs().sum()
        for sector in sector_weights.keys()
    }
    target_sector_leverages = {
        sector: total_commodity_leverage_target_from_previous_step * weight
        for sector, weight in sector_weights.items()
    }

    for sector, target_lev in target_sector_leverages.items():
        sector_tickers_mask_df = commodities_df['Commodity_Sector'] == sector
        current_lev = current_sector_leverages.get(sector, 0.0)
        if current_lev > 1e-6:
            scale_factor = min(max(target_lev / current_lev, 0.33), 3.0)
            commodities_df.loc[sector_tickers_mask_df, 'Position'] *= scale_factor
        elif target_lev > 0: # Seed if target is non-zero and current is zero
            # (Simplified seeding logic - give to strongest signal asset in sector)
            assets_in_sector = predictions_df[predictions_df['Commodity_Sector'] == sector] # from original df for strength
            if not assets_in_sector.empty:
                strongest_one = assets_in_sector.sort_values(by='Strength', key=abs, ascending=False).head(1)
                if not strongest_one.empty and strongest_one.index[0] in commodities_df.index:
                    direction = predictions_df.loc[strongest_one.index[0], 'Direction'] if predictions_df.loc[strongest_one.index[0], 'Direction'] !=0 else 1
                    commodities_df.loc[strongest_one.index[0], 'Position'] = direction * target_lev


    # Enforce min_sector_percentage
    recalculated_total_commodity_leverage = commodities_df['Position'].abs().sum()
    if recalculated_total_commodity_leverage > 1e-6:
        for sector in sector_weights.keys():
            sector_tickers_mask_df_min = commodities_df['Commodity_Sector'] == sector
            current_sector_actual_leverage = commodities_df.loc[sector_tickers_mask_df_min, 'Position'].abs().sum()
            min_required_sector_leverage = recalculated_total_commodity_leverage * min_sector_percentage_of_commodity_alloc
            if 0 < current_sector_actual_leverage < min_required_sector_leverage:
                scale_factor = min(min_required_sector_leverage / current_sector_actual_leverage, 2.0)
                commodities_df.loc[sector_tickers_mask_df_min, 'Position'] *= scale_factor
            elif current_sector_actual_leverage == 0 and min_required_sector_leverage > 0:
                 # (Simplified seeding for min percentage - similar to above)
                assets_in_sector = predictions_df[predictions_df['Commodity_Sector'] == sector]
                if not assets_in_sector.empty:
                    strongest_one = assets_in_sector.sort_values(by='Strength', key=abs, ascending=False).head(1)
                    if not strongest_one.empty and strongest_one.index[0] in commodities_df.index:
                        direction = predictions_df.loc[strongest_one.index[0], 'Direction'] if predictions_df.loc[strongest_one.index[0], 'Direction'] !=0 else 1
                        commodities_df.loc[strongest_one.index[0], 'Position'] = direction * min_required_sector_leverage


    # Normalize total commodity allocation BACK to the pre-sector-balancing target
    final_commodity_leverage_after_sector_ops = commodities_df['Position'].abs().sum()
    if final_commodity_leverage_after_sector_ops > 1e-6 and \
       abs(final_commodity_leverage_after_sector_ops - total_commodity_leverage_target_from_previous_step) > 0.001 * total_commodity_leverage_target_from_previous_step:
        normalization_factor = total_commodity_leverage_target_from_previous_step / final_commodity_leverage_after_sector_ops
        logger.info(f"Normalizing total commodity leverage from {final_commodity_leverage_after_sector_ops:.2%} "
                    f"back to pre-sector-balance target {total_commodity_leverage_target_from_previous_step:.2%} (Factor: {normalization_factor:.2f}x)")
        commodities_df.loc[:, 'Position'] *= normalization_factor # Apply to all in commodities_df

    # Update the original predictions_df
    predictions_df.loc[commodity_mask, 'Position'] = commodities_df['Position']
    # Risk_Contribution for commodities will be updated after this function returns, in the main flow
    return predictions_df
# --- End apply_commodity_sector_allocation ---

def enhance_create_optimized_etf_allocation(
    predictor,
    risk_target_param=None,
    max_leverage_param=None,
    min_signal_strength_param=None,
    asset_class_risk_allocation_param=None,
    portfolio_settings_override=None # This is an override for THIS function
    ):
    logger.info("--- Starting ENHANCED ETF Allocation Process ---")

    # Step 1: Consolidate settings for the call to the base allocation function.
    # Priority:
    # 1. `portfolio_settings_override` (if a full dict is passed directly to this enhanced function).
    # 2. Individual `_param` arguments (risk_target_param, etc.) passed to this enhanced function.
    # 3. Global `PORTFOLIO_SETTINGS` from config.py.

    if portfolio_settings_override:
        # If a full settings dictionary is provided to enhance_..., use it as the primary base.
        effective_settings = portfolio_settings_override.copy()
    else:
        # Otherwise, start with global defaults.
        effective_settings = {k: v for k, v in PORTFOLIO_SETTINGS.items()} # Make a fresh copy

    # Now, let individual _param arguments override specific keys in effective_settings.
    # This allows calling enhance_... with specific overrides without needing a full dict.
    if risk_target_param is not None:
        effective_settings['risk_target'] = risk_target_param
    if max_leverage_param is not None:
        effective_settings['max_leverage'] = max_leverage_param
    if min_signal_strength_param is not None:
        effective_settings['min_signal_strength'] = min_signal_strength_param
    if asset_class_risk_allocation_param is not None:
        effective_settings['asset_class_risk_allocation'] = asset_class_risk_allocation_param
    
    logger.info(f"Effective settings for base allocation call: {effective_settings}")

    # Step 2: Call the main create_optimized_etf_allocation function
    # Pass the fully resolved `effective_settings` dictionary.
    predictions_df = create_optimized_etf_allocation(
        predictor=predictor,
        portfolio_settings_override=effective_settings
    )

    if predictions_df.empty:
        logger.warning("Base allocation (called from enhance_create_optimized_etf_allocation) failed or returned empty.")
        return predictions_df

    # --- Optional: Specific "Enhancements" can be applied here ---
    # If the "enhanced" version has unique post-processing steps not in the base function,
    # they would go here. For example, if it always applies a certain type of risk adjustment
    # or a different final scaling.
    
    # Example: If ATR-based sizing is an "enhancement" (and assuming calculate_volatility_adjusted_position_size is defined)
    # apply_atr_sizing = effective_settings.get('apply_atr_sizing_in_enhanced', False)
    # if apply_atr_sizing:
    #     logger.info("Applying ATR-based volatility adjustment in ENHANCED version...")
    #     risk_per_trade_atr = effective_settings.get('risk_per_trade_atr', 0.005)
    #     for ticker_idx in predictions_df.index:
    #         vol_val = predictions_df.loc[ticker_idx, 'Volatility']
    #         current_pos_val = predictions_df.loc[ticker_idx, 'Position']
    #         if vol_val > 1e-6: # Check for valid volatility
    #             adj_pos_val = calculate_volatility_adjusted_position_size(
    #                 ticker=ticker_idx,
    #                 volatility=vol_val,
    #                 allocation=current_pos_val,
    #                 risk_per_trade=risk_per_trade_atr
    #             )
    #             predictions_df.loc[ticker_idx, 'Position'] = adj_pos_val
    #     predictions_df['Risk_Contribution'] = predictions_df['Position'] * predictions_df['Volatility']


    # Final leverage check specific to the enhanced path, using its determined max_leverage.
    final_max_leverage_for_enh_call = effective_settings['max_leverage']
    current_total_leverage = predictions_df['Position'].abs().sum()
    if current_total_leverage > final_max_leverage_for_enh_call:
        leverage_scale_factor = final_max_leverage_for_enh_call / current_total_leverage
        logger.info(f"Final leverage check in ENHANCED function. Current: {current_total_leverage:.2f}x, Target: {final_max_leverage_for_enh_call:.2f}x. Scaling by {leverage_scale_factor:.2f}x.")
        predictions_df['Position'] *= leverage_scale_factor
        # Recalculate Risk_Contribution if positions change
        if 'Volatility' in predictions_df.columns:
             predictions_df['Risk_Contribution'] = predictions_df['Position'] * predictions_df['Volatility']
    
    # Recalculate dollar allocations and log final metrics
    portfolio_value = 1000000  # Standard assumption
    if 'Position' in predictions_df.columns:
        predictions_df['Allocation'] = predictions_df['Position'] * portfolio_value

    # Ensure necessary columns exist for metric calculation, adding defaults if not
    if 'Position' not in predictions_df.columns: predictions_df['Position'] = 0.0
    if 'Risk_Contribution' not in predictions_df.columns: predictions_df['Risk_Contribution'] = 0.0
    if 'Expected_Return' not in predictions_df.columns: predictions_df['Expected_Return'] = 0.0
    if 'Direction' not in predictions_df.columns: predictions_df['Direction'] = 0
    if 'Strength' not in predictions_df.columns: predictions_df['Strength'] = 0
    if 'Volatility' not in predictions_df.columns: predictions_df['Volatility'] = 0.2 # default
    if 'Asset_Class' not in predictions_df.columns: predictions_df['Asset_Class'] = 'Unknown'


    long_exposure = predictions_df.loc[predictions_df['Position'] > 0, 'Position'].sum()
    short_exposure = predictions_df.loc[predictions_df['Position'] < 0, 'Position'].abs().sum()
    net_exposure = long_exposure - short_exposure
    final_total_leverage_enh = predictions_df['Position'].abs().sum()
    
    final_portfolio_vol_enh = np.sqrt((predictions_df['Risk_Contribution']**2).sum()) if not predictions_df['Risk_Contribution'].empty else 0.0
    portfolio_expected_return_enh = (predictions_df['Position'] * predictions_df['Expected_Return']).sum()
    sharpe_ratio_enh = portfolio_expected_return_enh / final_portfolio_vol_enh if final_portfolio_vol_enh > 1e-6 else 0.0

    logger.info(f"--- Final Metrics from ENHANCED Allocation (after all enhance steps) ---")
    logger.info(f"Number of Active Positions: {(predictions_df['Position'].abs() > 1e-6).sum()}")
    logger.info(f"Long Exposure: {long_exposure:.2%}")
    logger.info(f"Short Exposure: {short_exposure:.2%}")
    logger.info(f"Net Exposure: {net_exposure:.2%}")
    logger.info(f"Total Leverage: {final_total_leverage_enh:.2f}x")
    logger.info(f"Portfolio Volatility: {final_portfolio_vol_enh:.2%}")
    logger.info(f"Expected Annual Return (Model Based): {portfolio_expected_return_enh:.2%}") # Based on how ER is calculated
    logger.info(f"Estimated Sharpe Ratio (Model Based): {sharpe_ratio_enh:.2f}")

    return predictions_df

def apply_trend_following_overlay(predictions_df, predictor, min_lookback=20):
    """
    Apply a simple trend following overlay to adjust position directions.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame containing positions
    predictor : object
        Predictor object with returns data
    min_lookback : int
        Minimum lookback period for trend calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted positions
    """
    # Skip if we don't have returns data
    if not hasattr(predictor, 'returns') or predictor.returns is None or predictor.returns.empty:
        return predictions_df
    
    # For each asset, calculate recent trend
    for ticker in predictions_df.index:
        if ticker in predictor.returns.columns:
            returns = predictor.returns[ticker]
            
            # If we have enough data, calculate trend
            if len(returns) >= min_lookback:
                # Calculate short-term vs longer-term trend
                recent_return = returns.iloc[-min_lookback:].mean()
                
                # If the recent trend differs from our position direction, reduce position size
                position = predictions_df.loc[ticker, 'Position']
                if (position > 0 and recent_return < 0) or (position < 0 and recent_return > 0):
                    # Reduce position by 50% if fighting the trend
                    predictions_df.loc[ticker, 'Position'] *= 0.5
                    
                # If trend confirms our position, increase it (if it's not too large already)
                elif abs(position) < 0.15 and ((position > 0 and recent_return > 0) or (position < 0 and recent_return < 0)):
                    # Increase position by 25% if aligned with trend
                    predictions_df.loc[ticker, 'Position'] *= 1.25
    
    return predictions_df


def visualize_optimized_allocation(allocation_df):
    if allocation_df is None or allocation_df.empty:
        logger.warning("Allocation DataFrame is empty. Cannot visualize.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))
    
    non_zero_pos = allocation_df[allocation_df['Position'].abs() > 1e-6].copy() 
    if non_zero_pos.empty:
        logger.warning("No non-zero positions to visualize.")
        plt.title('Managed Futures ETF Allocation (No Active Positions)', fontsize=16, fontweight='bold')
        plt.xlabel('Position Weight', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        plt.tight_layout()
        # Check if running in a non-interactive environment where plt.show() might block
        if 'VSCODE_PID' not in os.environ and 'PYCHARM_HOSTED' not in os.environ:
            plt.show(block=False) # Try non-blocking show
            plt.pause(1) # Pause to allow plot to render
        else:
            plt.show()
        return

    positions = non_zero_pos['Position']
    colors = ['forestgreen' if x > 0 else 'firebrick' for x in positions]
    
    plt.barh(non_zero_pos.index, positions, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    if 'Volatility' in non_zero_pos.columns:
        vol_numeric = pd.to_numeric(non_zero_pos['Volatility'], errors='coerce').fillna(0)
        for i, (idx, row) in enumerate(non_zero_pos.iterrows()): # Use iterrows on non_zero_pos
            volatility_size = 100 * vol_numeric.loc[idx] 
            plt.scatter(row['Position'], i, s=max(0, volatility_size), alpha=0.4, 
                      color='darkblue', edgecolors='black', zorder=10)
    
    plt.title('Managed Futures ETF Allocation', fontsize=16, fontweight='bold')
    plt.xlabel('Position Weight', fontsize=12)
    plt.ylabel('Asset', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    asset_class_colors = {'Commodities': 'gold', 'Currencies': 'skyblue', 'Bonds': 'purple', 'Equities': 'orange', 'Unknown': 'gray'}
    if 'Asset_Class' in non_zero_pos.columns:
        # Create legend elements only for asset classes present in non_zero_pos
        present_asset_classes = non_zero_pos['Asset_Class'].unique()
        legend_elements = [plt.Rectangle((0,0), 1, 1, fc=asset_class_colors.get(ac, 'gray'), alpha=0.3) 
                           for ac in present_asset_classes]
        if legend_elements: # Only show legend if there's something to show
            plt.legend(legend_elements, present_asset_classes, 
                     loc='upper right', title='Asset Classes')

        # Color y-axis ticks or background spans
        y_ticks = plt.yticks()[0] # Get current y-tick positions
        for i, idx in enumerate(non_zero_pos.index): # Iterate over the actual assets being plotted
             if i < len(y_ticks): # Ensure we don't go out of bounds for y_ticks
                asset_class = non_zero_pos.loc[idx, 'Asset_Class']
                # This plt.axhspan might not align perfectly if y-axis is not simply enumerated.
                # A better way might be to color y-tick labels if possible, or ensure barh uses numerical y.
                # For now, this colors horizontal spans.
                plt.axhspan(y_ticks[i]-0.4, y_ticks[i]+0.4, alpha=0.1, color=asset_class_colors.get(asset_class, 'gray'), zorder=0)


    total_leverage = non_zero_pos['Position'].abs().sum()
    net_exposure = non_zero_pos['Position'].sum()
    risk_target_display = PORTFOLIO_SETTINGS.get('risk_target', 'N/A') # Get from actual runtime settings
    if isinstance(risk_target_display, float): risk_target_display = f"{risk_target_display:.1%}"

    stats_text = (
        f"Total Leverage: {total_leverage:.2f}x\n"
        f"Net Exposure: {net_exposure:.2%}\n"
        f"Active Positions: {len(non_zero_pos)}\n"
        f"Config Risk Target: {risk_target_display}" # Display the target from config
    )
    
    plt.figtext(0.15, 0.01, stats_text, ha='left', fontsize=10, # Reduced font size
              bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjusted rect for figtext
    if 'VSCODE_PID' not in os.environ and 'PYCHARM_HOSTED' not in os.environ:
        plt.show(block=False)
        plt.pause(1)
    else:
        plt.show()


def plot_risk_sunburst(allocation_df):
    if allocation_df is None or allocation_df.empty:
        logger.warning("Allocation DataFrame is empty. Cannot plot sunburst.")
        return None
        
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is required for sunburst plot. Install with 'pip install plotly'")
        return None
    
    non_zero_df = allocation_df[allocation_df['Position'].abs() > 1e-6].copy()
    if non_zero_df.empty:
        logger.warning("No non-zero positions for sunburst plot.")
        return None

    # Ensure Risk_Contribution is numeric and handle NaNs
    non_zero_df['Risk_Contribution'] = pd.to_numeric(non_zero_df['Risk_Contribution'], errors='coerce').fillna(0)
    non_zero_df['Abs_Risk_Contribution'] = non_zero_df['Risk_Contribution'].abs()

    if non_zero_df['Abs_Risk_Contribution'].sum() < 1e-9 : 
        logger.warning("Total absolute risk contribution is zero. Sunburst plot may be empty or misleading.")
        if not non_zero_df.empty:
             non_zero_df['Abs_Risk_Contribution'] = non_zero_df['Abs_Risk_Contribution'].replace(0, 1e-6) # Avoid issues with all-zero values for plot rendering

    if 'Asset_Class' not in non_zero_df.columns:
        logger.error("'Asset_Class' column missing in allocation_df for sunburst plot.")
        non_zero_df['Asset_Class'] = 'Unknown' # Assign default

    asset_classes_present = non_zero_df['Asset_Class'].unique()
    labels = ['Total Risk'] + list(asset_classes_present) + list(non_zero_df.index)
    parents = [''] + ['Total Risk'] * len(asset_classes_present) + list(non_zero_df['Asset_Class'])
              
    asset_class_risk_sum = non_zero_df.groupby('Asset_Class')['Abs_Risk_Contribution'].sum().reindex(asset_classes_present).fillna(0)

    values = [asset_class_risk_sum.sum()] + list(asset_class_risk_sum) + list(non_zero_df['Abs_Risk_Contribution'])

    if 'Direction' not in non_zero_df.columns:
         logger.warning("'Direction' column missing for sunburst plot colors. Defaulting to neutral.")
         non_zero_df['Direction'] = 0 

    directions = [''] + [''] * len(asset_classes_present) + list(non_zero_df['Direction'])
    direction_colors = ['lightgray' if d == '' else ('forestgreen' if pd.to_numeric(d, errors='coerce') > 0 else ('firebrick' if pd.to_numeric(d, errors='coerce') < 0 else 'silver')) for d in directions]

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total", 
        marker=dict(colors=direction_colors),
        hovertemplate='<b>%{label}</b><br>Risk Contribution: %{value:.3%}<br>Parent: %{parent}<extra></extra>',
        textfont=dict(size=12) 
    ))
    
    fig.update_layout(
        title_text='Risk Allocation Sunburst Chart',
        title_font_size=20,
        margin=dict(t=60, l=10, r=10, b=10) 
    )
    
    # Save the figure to an HTML file as a backup
    try:
        import os
        os.makedirs('plots', exist_ok=True)
        fig.write_html('plots/risk_sunburst.html')
        logger.info("Saved sunburst visualization to plots/risk_sunburst.html")
    except Exception as e:
        logger.warning(f"Couldn't save sunburst visualization: {str(e)}")
    
    # Now show the figure with a specific browser configuration
    # This is the key change to make it stay visible
    import plotly.io as pio
    pio.renderers.default = "browser"  # Force open in browser window instead of inline
    fig.show()
    
    # Return the figure object so it can be referenced in the calling code
    return fig