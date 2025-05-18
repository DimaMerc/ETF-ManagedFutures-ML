# rebalancing.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RebalanceTracker:
    """
    Tracks rebalancing schedule and determines when rebalancing should occur.
    """
    def __init__(self, config=None):
        """
        Initialize the rebalance tracker.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.quarterly_rebalance = self.config.get('enable_quarterly_rebalance', True)
        self.signal_driven_rebalance = self.config.get('enable_signal_driven_rebalance', True)
        self.signal_threshold = self.config.get('signal_change_threshold', 0.15)
        self.min_days_between_rebalances = self.config.get('min_days_between_rebalances', 14)
        
        # Tracking variables
        self.last_rebalance_date = None
        self.previous_signals = None
        self.previous_economic_regime = None
        
        logger.info(f"Initialized RebalanceTracker with parameters: "
                   f"quarterly={self.quarterly_rebalance}, "
                   f"signal_driven={self.signal_driven_rebalance}, "
                   f"threshold={self.signal_threshold}, "
                   f"min_days={self.min_days_between_rebalances}")
    
    def is_quarterly_boundary(self, current_date=None):
        """
        Check if current date is at a quarterly boundary.
        
        Parameters:
        -----------
        current_date : datetime, optional
            Current date (defaults to today)
            
        Returns:
        --------
        bool
            True if at quarterly boundary
        """
        if not self.quarterly_rebalance:
            return False
            
        current_date = current_date or datetime.now()
        
        # First day of the month check
        if current_date.day != 1:
            return False
            
        # Quarter month check (Jan, Apr, Jul, Oct)
        if current_date.month not in [1, 4, 7, 10]:
            return False
            
        return True
    
    def should_rebalance(self, predictor, current_date=None):
        """
        Determine if a rebalance should happen based on schedule and signals.
        
        Parameters:
        -----------
        predictor : CudaOptimizedTrendPredictor
            Predictor with current signals
        current_date : datetime, optional
            Current date (defaults to today)
            
        Returns:
        --------
        bool
            True if rebalance should happen
        """
        current_date = current_date or datetime.now()
        
        # Get current signals
        current_signals = None
        if hasattr(predictor, 'final_direction') and predictor.final_direction is not None:
            current_signals = predictor.final_direction * predictor.final_strength
        elif hasattr(predictor, 'adjusted_direction') and predictor.adjusted_direction is not None:
            current_signals = predictor.adjusted_direction * predictor.adjusted_strength
        elif hasattr(predictor, 'trend_direction') and predictor.trend_direction is not None:
            current_signals = predictor.trend_direction * predictor.trend_strength
        
        if current_signals is None:
            logger.warning("No valid signals available for rebalance check")
            return False
        
        # Minimum time between rebalances check
        if self.last_rebalance_date is not None:
            days_since_last = (current_date - self.last_rebalance_date).days
            if days_since_last < self.min_days_between_rebalances:
                logger.info(f"Only {days_since_last} days since last rebalance. "
                           f"Minimum is {self.min_days_between_rebalances}.")
                return False
        
        # Quarterly schedule check
        if self.is_quarterly_boundary(current_date):
            logger.info("Rebalance triggered by quarterly schedule")
            return True
        
        # If signal-driven rebalancing is disabled, stop here
        if not self.signal_driven_rebalance:
            return False
            
        # If no previous signals, we can't compare
        if self.previous_signals is None or len(self.previous_signals) == 0:
            logger.info("No previous signals available for comparison. Skipping signal check.")
            return False
        
        # 1. Check for direction reversals in major assets
        major_assets = ['SPY', 'TLT', 'BRENT', 'NATURAL_GAS', 'IWM']
        for asset in major_assets:
            if asset in current_signals.index and asset in self.previous_signals.index:
                current_sign = np.sign(current_signals[asset])
                previous_sign = np.sign(self.previous_signals[asset])
                if current_sign != 0 and previous_sign != 0 and current_sign != previous_sign:
                    logger.info(f"Rebalance triggered by direction reversal in {asset}")
                    return True
        
        # 2. Check for significant magnitude changes across all signals
        common_assets = set(current_signals.index).intersection(self.previous_signals.index)
        if common_assets:
            common_assets = list(common_assets)
            signal_changes = (current_signals[common_assets] - self.previous_signals[common_assets]).abs()
            max_change = signal_changes.max()
            avg_change = signal_changes.mean()
            
            if max_change > self.signal_threshold:
                logger.info(f"Rebalance triggered by large signal change: {max_change:.2f} > {self.signal_threshold:.2f}")
                return True
            if avg_change > self.signal_threshold / 2:
                logger.info(f"Rebalance triggered by significant average signal change: {avg_change:.2f}")
                return True
        
        # 3. Check for economic regime changes
        current_regime = getattr(predictor, 'economic_regime', None)
        if current_regime is not None and self.previous_economic_regime is not None:
            if current_regime != self.previous_economic_regime:
                logger.info(f"Rebalance triggered by economic regime change: "
                          f"{self.previous_economic_regime} → {current_regime}")
                return True
        
        # No need to rebalance
        logger.info("No rebalance triggers detected")
        return False
    
    def update_tracking(self, predictor, current_date=None):
        """
        Update tracking variables after a rebalance.
        
        Parameters:
        -----------
        predictor : CudaOptimizedTrendPredictor
            Predictor with current signals
        current_date : datetime, optional
            Current date (defaults to today)
        """
        current_date = current_date or datetime.now()
        self.last_rebalance_date = current_date
        
        # Store current signals for next comparison
        if hasattr(predictor, 'final_direction') and predictor.final_direction is not None:
            self.previous_signals = predictor.final_direction * predictor.final_strength
        elif hasattr(predictor, 'adjusted_direction') and predictor.adjusted_direction is not None:
            self.previous_signals = predictor.adjusted_direction * predictor.adjusted_strength
        elif hasattr(predictor, 'trend_direction') and predictor.trend_direction is not None:
            self.previous_signals = predictor.trend_direction * predictor.trend_strength
        
        # Store economic regime
        self.previous_economic_regime = getattr(predictor, 'economic_regime', None)
        
        logger.info(f"Updated rebalance tracking. Next rebalance will be after {current_date + timedelta(days=self.min_days_between_rebalances)}")