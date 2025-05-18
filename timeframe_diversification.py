# timeframe_diversification.py
"""
Implements a simplified timeframe diversification concept for the trend prediction system.
It weights the base predictor's signals based on conceptual timeframes and market regime.
A true multi-timeframe system would involve models trained on data of different resolutions.
"""

import numpy as np
import pandas as pd
import torch # Keep for type hints if CudaOptimizedTrendPredictor returns tensors
import logging
# from simple_lstm_model import SimpleLSTM # Not needed if not building new models here

logger = logging.getLogger(__name__)

# timeframe_diversification.py

def detect_market_regime(returns_data, window=63, market_index_symbol='SPY'):
    # ... (market_index_to_use selection as before) ...
    if market_index_symbol not in returns_data.columns:
        logger.warning(f"Market index '{market_index_symbol}' not in returns data for regime. Defaulting to neutral.")
        return 'neutral'

    market_returns_series = returns_data[market_index_symbol].dropna()
    # V8 FIX: Check against the full window length needed for a stable calculation
    if len(market_returns_series) < window: 
        logger.warning(f"Not enough data for market index '{market_index_symbol}' (have {len(market_returns_series)}, need {window}). Defaulting to neutral regime.")
        return 'neutral'
    
    market_returns = market_returns_series.iloc[-window:] # Now this slice is safe
    # ... (rest of regime detection logic using market_returns) ...
    log_returns = np.log(1 + market_returns) # market_returns is now guaranteed to have 'window' points
    cumulative_log_return = log_returns.sum()
    approx_cumulative_pct_return = np.exp(cumulative_log_return) - 1

    if approx_cumulative_pct_return > 0.03: return 'bull'
    elif approx_cumulative_pct_return < -0.03: return 'bear'
    else: return 'neutral'

# The rest of TimeframeDiversifiedPredictor class from previous response is okay for this fix.
# It correctly uses the base_predictor's signals and re-weights them.
# The main problem was the key mismatch ('5' vs '5d').

class TimeframeDiversifiedPredictor:
    """
    Applies timeframe-based weighting to signals from a base predictor.
    This is a simplified approach. A full multi-timeframe system
    would train distinct models on data resampled to different frequencies.
    """
    
    def __init__(self, base_predictor, timeframes=None, regime_weights=None):
        """
        Initialize timeframe diversified predictor.
        
        Parameters:
        -----------
        base_predictor : CudaOptimizedTrendPredictor
            The main predictor instance whose signals will be re-weighted.
        timeframes : list, optional
            List of conceptual timeframes (e.g., ['short', 'medium', 'long'] or [5, 20, 60]).
            The actual numeric values will map to keys in regime_weights.
        regime_weights : dict, optional
            Weights for each timeframe under different market regimes.
            Example: {'bull': {'5d': 0.4, '20d': 0.4, '60d': 0.2}, ...}
        """
        self.predictor = base_predictor # This is the CudaOptimizedTrendPredictor instance
        
        # Use default timeframes as keys if none provided
        self.timeframes = timeframes or ['5d', '20d', '60d'] # Conceptual timeframes

        if regime_weights is None:
            self.regime_weights = {
                'bull':    {'5d': 0.5, '20d': 0.3, '60d': 0.2}, # Emphasize shorter term in bull
                'bear':    {'5d': 0.2, '20d': 0.3, '60d': 0.5}, # Emphasize longer term in bear
                'neutral': {'5d': 0.3, '20d': 0.4, '60d': 0.3}  # Balanced in neutral
            }
        else:
            self.regime_weights = regime_weights
            
        logger.info(f"Initializing TimeframeDiversifiedPredictor with conceptual timeframes: {self.timeframes}")
        # No separate models are built or stored in this simplified version.
        # We rely entirely on the base_predictor's trained models.

    def _get_base_predictions(self):
        """
        Ensures the base predictor has generated its predictions.
        Returns the Direction and Strength series.
        """
        if not (hasattr(self.predictor, 'final_direction') and \
                  self.predictor.final_direction is not None and \
                  hasattr(self.predictor, 'final_strength') and \
                  self.predictor.final_strength is not None):
            logger.warning("Base predictor has no final_direction/strength. Attempting to generate them.")
            # This assumes predict_trends_pipeline is a method of the base predictor
            # or a global function that takes the predictor.
            # For this example, let's assume it's a method.
            if hasattr(self.predictor, 'predict_trends_pipeline'):
                 self.predictor.predict_trends_pipeline() # This will run base predict + economic/news adjustments
            elif hasattr(self.predictor, 'predict_trends'): # Fallback to simpler prediction call
                 self.predictor.predict_trends()
                 self.predictor.adjust_for_economic_uncertainty()
                 self.predictor.adjust_for_news_sentiment()
            else:
                logger.error("Base predictor does not have a suitable prediction generation method.")
                return None, None

        # Use the most refined signals available from the base predictor
        if hasattr(self.predictor, 'final_direction') and self.predictor.final_direction is not None:
            return self.predictor.final_direction, self.predictor.final_strength
        elif hasattr(self.predictor, 'adjusted_direction') and self.predictor.adjusted_direction is not None:
            return self.predictor.adjusted_direction, self.predictor.adjusted_strength
        elif hasattr(self.predictor, 'trend_direction') and self.predictor.trend_direction is not None:
            return self.predictor.trend_direction, self.predictor.trend_strength
        else:
            logger.error("Base predictor failed to produce any usable signals.")
            return None, None


    def predict_with_timeframe_diversification(self):
        """
        Generate combined predictions by re-weighting the base predictor's signals
        according to conceptual timeframes and market regime.

        Returns:
        --------
        pd.Series
            Combined 'Position' signals (Direction * Strength * TimeframeWeight)
            The names of assets will be the index.
        """
        logger.info("--- Applying Timeframe Diversification (Simplified Re-weighting) ---")

        base_directions, base_strengths = self._get_base_predictions()

        if base_directions is None or base_strengths is None:
            logger.error("Could not get base predictions. Timeframe diversification cannot proceed.")
            # Return empty series with same index as predictor's assets if possible, else None
            if hasattr(self.predictor, 'returns') and self.predictor.returns is not None:
                return pd.Series(dtype=float, index=self.predictor.returns.columns)
            return None

        # Detect current market regime using the base predictor's returns data
        # Ensure self.predictor.returns is populated and is a DataFrame
        market_returns_data = None
        if hasattr(self.predictor, 'returns') and isinstance(self.predictor.returns, pd.DataFrame) and not self.predictor.returns.empty:
            market_returns_data = self.predictor.returns
        else:
            logger.warning("Base predictor's returns data not available or not a DataFrame. Cannot detect market regime. Defaulting to neutral.")
        
        regime = detect_market_regime(market_returns_data) if market_returns_data is not None else 'neutral'
        logger.info(f"Detected market regime for timeframe weighting: {regime}")
        
        # Get timeframe weights for the current regime
        current_timeframe_weights = self.regime_weights.get(regime, self.regime_weights['neutral'])
        logger.info(f"Using timeframe weights for {regime}: {current_timeframe_weights}")

        # Initialize combined signals (these will be the final 'Position' like signals)
        # The index should cover all assets for which the base predictor might have signals.
        all_asset_tickers = base_directions.index
        combined_signal_values = pd.Series(0.0, index=all_asset_tickers, dtype=float)

        # Apply weights
        # In this simplified approach, each "conceptual timeframe" essentially gets a weight,
        # and we average the base signal scaled by these weights.
        # This means we are not using separate models, but re-weighting a single set of signals.
        
        total_weight_applied = 0.0
        for timeframe_key in self.timeframes: # e.g., '5d', '20d', '60d'
            weight = current_timeframe_weights.get(timeframe_key, 0.0) # Get weight for this conceptual timeframe
            if weight == 0.0:
                logger.info(f"  Skipping timeframe '{timeframe_key}' due to zero weight in current regime '{regime}'.")
                continue
            
            total_weight_applied += weight
            # For each asset, add its base signal (Direction * Strength) multiplied by this timeframe's weight
            for ticker in all_asset_tickers:
                if ticker in base_directions.index and ticker in base_strengths.index:
                    signal_contribution = base_directions[ticker] * base_strengths[ticker] * weight
                    combined_signal_values[ticker] += signal_contribution
                # else: No base signal for this ticker, so it gets no contribution from this timeframe

        # Normalize if total_weight_applied is not 1.0 (it should be if weights sum to 1 per regime)
        if total_weight_applied > 0 and abs(total_weight_applied - 1.0) > 1e-6:
            logger.info(f"Normalizing combined signals as total timeframe weight was {total_weight_applied:.2f}")
            combined_signal_values /= total_weight_applied
            
        # The combined_signal_values now represent the re-weighted "Position" signals
        # The main CudaOptimizedTrendPredictor will then use these as its new
        # trend_predictions, from which trend_direction and trend_strength are derived.

        logger.info(f"Generated combined (re-weighted) signals for {len(combined_signal_values[combined_signal_values.abs() > 1e-6])} assets.")
        
        # For debugging, log top signals
        # top_signals = combined_signal_values.abs().sort_values(ascending=False).head(5)
        # for ticker_log, val_log in top_signals.items():
        #     logger.info(f"  Top combined signal: {ticker_log} = {combined_signal_values[ticker_log]:.4f}")

        return combined_signal_values # This Series is effectively the new 'trend_predictions'