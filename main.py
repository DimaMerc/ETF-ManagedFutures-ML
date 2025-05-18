# main.py

"""
Optimized main.py with enhanced CUDA trend prediction pipeline
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
# from typing import Dict, List, Tuple, Optional, Union # Can remove if not explicitly used for type hints


# Import optimized components
from cuda_optimized_trend_predictor import CudaOptimizedTrendPredictor
from alpha_vantage_client_optimized import AlphaVantageClient, CommodityData
from lstm_model import OptimizedLSTM, EnsembleTrendModel, TrendPredictionModule # Assuming these are used by predictor
# Ensure the correct functions are imported from allocation_utils
from allocation_utils import (
    create_optimized_etf_allocation, 
    enhance_create_optimized_etf_allocation,
    visualize_optimized_allocation, # Assuming this is now in allocation_utils
    plot_risk_sunburst             # Assuming this is now in allocation_utils
)
from error_handlers_and_utils import (
    NotEnoughDataError, # Assuming this is used by predictor or other utils
    # augment_time_series, time_warp, magnitude_scale, add_financial_noise # If used directly in main
)
# Assuming visualization_utils still holds some specific visualization functions
from visualization_utils import (
    create_prediction_plots, 
    # create_allocation_plots, # This might now be in allocation_utils or called from there
    compare_portfolio_allocations, 
    plot_commodity_sector_allocation,
    create_alpha_vantage_etf_comparison,
    display_benchmark_metrics
)


# Import configuration
from config import ASSETS, MODEL_PARAMS, CUDA_SETTINGS, PORTFOLIO_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etf_strategy.log", mode='w'), # mode='w' to overwrite log each run
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Import rebalancing tracker
try:
    from rebalancing import RebalanceTracker
except ImportError:
    logger.warning("rebalancing.py not found. Rebalancing features will be disabled.")
    RebalanceTracker = None


def parse_arguments():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(description='Financial Trend Prediction with CUDA Optimizations')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Path to financial data directory')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of ticker symbols (overrides config.py)')
    parser.add_argument('--start_date', type=str, default='2010-01-01',
                        help='Start date for data analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data analysis (YYYY-MM-DD)')
    
    # Model parameters
    parser.add_argument('--sequence_length', type=int, default=None,
                        help='Sequence length for time series analysis (overrides config.py)')
    parser.add_argument('--hidden_dim', type=int, default=None, 
                        help='Base hidden dimension for LSTM layers (overrides config.py)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of LSTM layers (overrides config.py)')
    
    # Training parameters
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--load_model', action='store_true',
                        help='Load a pre-trained model')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config.py)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config.py)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate for optimizer (overrides config.py)')
    parser.add_argument('--force_rebuild', action='store_true', 
                        help='Force rebuild of models even if they already exist')
    
    # Prediction parameters
    parser.add_argument('--predict', action='store_true',
                        help='Generate predictions')
    
    # Timeframe diversification parameters
    parser.add_argument('--enable_timeframe_diversification', action='store_true',
                    help='Enable multi-timeframe prediction diversification')
    
    # Allocation parameters
    parser.add_argument('--allocate', action='store_true',
                        help='Generate portfolio allocation')
    parser.add_argument('--risk_target', type=float, default=None,
                        help='Target portfolio risk (annualized volatility, overrides config.py)')
    parser.add_argument('--max_leverage', type=float, default=None,
                        help='Maximum portfolio leverage (overrides config.py)')
    parser.add_argument('--min_signal_strength', type=float, default=None,
                        help='Minimum signal strength for position (overrides config.py)')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--show_attention', action='store_true',
                        help='Visualize attention weights in the model')
    parser.add_argument('--interactive_plots', action='store_true', # Renamed for clarity
                        help='Generate interactive visualizations (requires plotly)')
    
    parser.add_argument('--compare_etfs', action='store_true',
                    help='Compare strategy against ETF benchmarks and indices')
    parser.add_argument('--comparison_period', type=int, default=365,
                    help='Number of days for ETF performance comparison (default: 365)')
        
    # CUDA optimization parameters
    parser.add_argument('--cuda', action='store_true', 
                        help='Use CUDA acceleration (automatically detected if available)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (FP16/FP32) for faster computation')
    parser.add_argument('--use_cuda_graphs', action='store_true',
                        help='Use CUDA Graphs for optimized GPU execution (RTX 5090)')
    parser.add_argument('--tensor_cores', action='store_true',
                        help='Explicitly enable Tensor Cores on compatible GPUs')
    parser.add_argument('--rtx_optimized', action='store_true',
                        help='Apply specific optimizations for RTX 5090 GPUs')
    
    # Dashboard parameters
    parser.add_argument('--dashboard', action='store_true',
                        help='Run the ETF dashboard application')
    
    # Benchmark parameters
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmarks')
    
    return parser.parse_args()


def update_config_from_args(args):
    updated_model_params = {k: v for k, v in MODEL_PARAMS.items()}
    updated_cuda_settings = {k: v for k, v in CUDA_SETTINGS.items()}
    updated_portfolio_settings = {k: v for k, v in PORTFOLIO_SETTINGS.items()}
    
    if args.sequence_length is not None: updated_model_params['sequence_length'] = args.sequence_length
    if args.num_layers is not None: updated_model_params['num_layers'] = args.num_layers
    
    if args.hidden_dim is not None:
        num_layers_eff = args.num_layers if args.num_layers is not None else updated_model_params.get('num_layers', 3)
        # Create a geometrically decreasing sequence of hidden sizes if hidden_dim is given
        updated_model_params['hidden_sizes'] = [max(16, int(args.hidden_dim / (2**i))) for i in range(num_layers_eff)]
        if not updated_model_params['hidden_sizes']: # Ensure it's not empty
             updated_model_params['hidden_sizes'] = [args.hidden_dim]


    if args.epochs is not None: updated_model_params['epochs'] = args.epochs
    if args.batch_size is not None: updated_model_params['batch_size'] = args.batch_size
    if args.learning_rate is not None: updated_model_params['learning_rate'] = args.learning_rate
        
    updated_cuda_settings['enable_cuda'] = torch.cuda.is_available() and (args.cuda or CUDA_SETTINGS.get('enable_cuda', True))
    if args.mixed_precision is not None: updated_cuda_settings['enable_mixed_precision'] = args.mixed_precision
    if args.use_cuda_graphs is not None: updated_cuda_settings['enable_cuda_graphs'] = args.use_cuda_graphs
    if args.tensor_cores is not None: updated_cuda_settings['use_tensor_cores'] = args.tensor_cores
    if args.rtx_optimized:
        updated_cuda_settings.update({
            'enable_mixed_precision': True, 'enable_cuda_graphs': True, 'use_tensor_cores': True,
            'set_float32_precision': 'high', 'enable_cudnn_benchmark': True,
        })
        logger.info("Applied RTX optimizations")
    
    if args.risk_target is not None: updated_portfolio_settings['risk_target'] = args.risk_target
    if args.max_leverage is not None: updated_portfolio_settings['max_leverage'] = args.max_leverage
    if args.min_signal_strength is not None: updated_portfolio_settings['min_signal_strength'] = args.min_signal_strength
    
    # Ensure commodity target leverage in portfolio settings is updated if risk allocation for commodities changes
    # This assumes asset_class_risk_allocation for commodities is used as a direct leverage target for them.
    if 'asset_class_risk_allocation' in updated_portfolio_settings and \
       'Commodities' in updated_portfolio_settings['asset_class_risk_allocation']:
        updated_portfolio_settings['commodity_target_leverage'] = updated_portfolio_settings['asset_class_risk_allocation']['Commodities']
  
    # Default rebalancing settings
    if 'rebalancing' not in updated_portfolio_settings:
        updated_portfolio_settings['rebalancing'] = {
            'enable_quarterly_rebalance': True,
            'enable_signal_driven_rebalance': True,
            'signal_change_threshold': 0.15,
            'min_days_between_rebalances': 14,
            'force_initial_allocation': True
        }
    

    return updated_model_params, updated_cuda_settings, updated_portfolio_settings


def get_assets_dict(args):
    if args.symbols:
        symbols_list = args.symbols.split(',')
        assets_dict = {'Commodities': [], 'Currencies': [], 'Bonds': [], 'Equities': [], 'Unknown': []}
        for symbol in symbols_list:
            symbol_upper = symbol.upper()
            if symbol_upper in ['BRENT','WTI','NATURAL_GAS','COPPER','ALUMINUM','WHEAT','CORN', 'XAU', 'XAG', 'OIL', 'DBA', 'USO', 'GLD']:
                assets_dict['Commodities'].append(symbol_upper)
            elif symbol_upper in ['EUR','GBP','JPY','CAD','AUD','CHF','NZD', 'USD', 'UUP', 'FXE'] or '/' in symbol: 
                assets_dict['Currencies'].append(symbol_upper)
            elif symbol_upper in ['TLT','IEF','SHY','GOVT','BND', 'AGG', 'LQD', 'HYG', 'TIP']: 
                assets_dict['Bonds'].append(symbol_upper)
            elif len(symbol_upper) <= 5 and symbol_upper.isalpha() and symbol_upper not in assets_dict['Commodities'] and symbol_upper not in assets_dict['Currencies'] and symbol_upper not in assets_dict['Bonds']: # Avoid re-adding
                 assets_dict['Equities'].append(symbol_upper)
            else: 
                if symbol_upper not in assets_dict['Commodities'] and symbol_upper not in assets_dict['Currencies'] and symbol_upper not in assets_dict['Bonds'] and symbol_upper not in assets_dict['Equities']:
                    assets_dict['Unknown'].append(symbol_upper)
        return {k: v for k, v in assets_dict.items() if v}
    else:
        return ASSETS


def preprocess_data_for_training(args):
    """
    Enhance data quality before training models.
    """
    # Get ALPHA_VANTAGE_API_KEY from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY not set. Cannot fetch data.")
        return None
    
    # Create AlphaVantage client
    av_client = AlphaVantageClient(api_key)
    
    # Get assets dictionary from args or config
    assets_dict = get_assets_dict(args)
    
    # Create mapping of ETF proxies for assets with limited data
    etf_proxies = {
        'BRENT': 'BNO',         # Brent Oil ETF
        'WTI': 'USO',           # WTI Oil ETF
        'NATURAL_GAS': 'UNG',   # Natural Gas ETF 
        'COPPER': 'CPER',       # Copper ETF
        'ALUMINUM': 'JJU',      # Aluminum ETF
        'WHEAT': 'WEAT',        # Wheat ETF
        'CORN': 'CORN',         # Corn ETF
    }
    
    # Create a dictionary to store enhanced data
    enhanced_returns = {}
    
    # Process each asset class
    for asset_class, tickers in assets_dict.items():
        logger.info(f"Preprocessing data for {asset_class} assets...")
        
        for ticker in tickers:
            try:
                # Try to get extended data (1460 days instead of default)
                if asset_class == 'Commodities':
                    # Try ETF proxy first (they have more reliable data)
                    if ticker in etf_proxies:
                        proxy = etf_proxies[ticker]
                        logger.info(f"Using ETF proxy {proxy} for {ticker}")
                        df = av_client.get_daily_adjusted(proxy, outputsize="full", cache=True)
                        
                        if not df.empty and len(df) > 60:  # Check if we got enough data
                            enhanced_returns[ticker] = df['returns']
                            logger.info(f"Successfully loaded {len(df)} days of proxy data for {ticker}")
                            continue
                    
                    # If proxy failed or not available, try direct commodity data
                    df = av_client.commodity_client.get_commodity_data(
                        ticker, interval='daily', cache=True)
                elif asset_class == 'Currencies':
                    # Handle currency pairs
                    if '/' in ticker:
                        from_curr, to_curr = ticker.split('/')
                    else:
                        if ticker in ['JPY', 'CAD', 'CHF']:
                            from_curr, to_curr = 'USD', ticker
                        else:
                            from_curr, to_curr = ticker, 'USD'
                    
                    df = av_client.get_fx_rate(from_curr, to_curr, interval='daily', cache=True)
                else:
                    # For bonds and equities
                    df = av_client.get_daily_adjusted(ticker, outputsize="full", cache=True)
                
                # Check if we got enough data
                if df is not None and not df.empty:
                    # Find the returns column
                    if 'returns' in df.columns:
                        returns_col = 'returns'
                    elif df.shape[0] > 1:
                        # Try to find price column to calculate returns
                        for col in ['close', 'adjusted_close', 'value', 'price']:
                            if col in df.columns:
                                df['returns'] = df[col].pct_change()
                                returns_col = 'returns'
                                break
                    
                    if returns_col:
                        enhanced_returns[ticker] = df[returns_col]
                        logger.info(f"Successfully loaded {len(df)} days of data for {ticker}")
                    else:
                        logger.warning(f"Could not find or calculate returns for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")
            
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
    
    # Convert to DataFrame
    if enhanced_returns:
        returns_df = pd.DataFrame(enhanced_returns)
        
        # Save to cache file for later use
        cache_file = "enhanced_returns_data.csv"
        returns_df.to_csv(cache_file)
        logger.info(f"Saved enhanced returns data for {len(returns_df.columns)} assets")
        
        return returns_df
    else:
        logger.error("Failed to preprocess any data")
        return None

def train_models_pipeline(predictor, args): 
    logger.info("Starting model training pipeline...")
    try:
        logger.info("Loading returns data...")
        predictor.load_returns_data()
        logger.info("Loading technical indicators (using potentially enhanced method)...")
        predictor.load_technical_indicators() 
        logger.info("Loading economic data...")
        predictor.load_economic_data()
        logger.info("Loading news sentiment...")
        predictor.load_news_sentiment()
        
        logger.info("Building trend models...")
        predictor.build_trend_models(force_rebuild=args.force_rebuild)
        logger.info("Saving trend models...")
        predictor.save_trend_models()
        logger.info("Training pipeline completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during model training pipeline: {str(e)}", exc_info=True)
        return False


def predict_trends_pipeline(predictor): 
    try:
        logger.info("Generating trend predictions pipeline...")
        trend_predictions, _, _ = predictor.predict_trends()
        if trend_predictions is None: 
            logger.error("Base trend prediction failed.")
            return False

        logger.info("Adjusting for economic uncertainty...")
        predictor.adjust_for_economic_uncertainty()

        logger.info("Adjusting for news sentiment...")
        final_direction, _ = predictor.adjust_for_news_sentiment() # Get final_direction
        if final_direction is None:
            logger.warning("News sentiment adjustment failed.")
            return False
        
        logger.info(f"Generated final trend predictions for {len(final_direction)} assets")
        return True
    except Exception as e:
        logger.error(f"Error generating predictions pipeline: {str(e)}", exc_info=True)
        return False


def generate_allocation_pipeline(predictor, portfolio_settings_config, use_enhanced_allocation=True, rebalance_tracker=None):
    try:
        current_date = datetime.now()
        
        # Check if rebalancing is needed (if tracker is available)
        if rebalance_tracker is not None:
            should_rebalance = rebalance_tracker.should_rebalance(predictor, current_date)
            
            # If no rebalance needed and we have a previous allocation, use that
            if not should_rebalance and hasattr(rebalance_tracker, 'last_allocation'):
                logger.info(f"No rebalance needed on {current_date}. Using previous allocation.")
                return rebalance_tracker.last_allocation
            
            logger.info(f"Rebalance required on {current_date}. Generating new allocation.")
        
        # Continue with standard allocation process
        if not hasattr(predictor, 'final_direction') or predictor.final_direction is None:
            logger.warning("No final predictions. Running prediction pipeline first.")
            if not predict_trends_pipeline(predictor): 
                 logger.error("Prediction pipeline failed. Cannot generate allocation.")
                 return None
        
        logger.info(f"Generating {'ENHANCED' if use_enhanced_allocation else 'STANDARD'} ETF allocation...")
        if use_enhanced_allocation:
            allocation = enhance_create_optimized_etf_allocation(
                predictor=predictor,
                portfolio_settings_override=portfolio_settings_config
            )
            filename = 'enhanced_etf_allocation.csv'
        else:
            allocation = create_optimized_etf_allocation(
                predictor=predictor,
                portfolio_settings_override=portfolio_settings_config 
            )
            filename = 'standard_etf_allocation.csv'

        if allocation is None or allocation.empty:
            logger.error("Allocation generation failed or returned empty.")
            return None

        allocation.to_csv(filename)
        logger.info(f"Allocation exported to {filename}")
        
        # Update rebalance tracker if available
        if rebalance_tracker is not None:
            rebalance_tracker.update_tracking(predictor, current_date)
            rebalance_tracker.last_allocation = allocation.copy()  # Store for future reference
            
            # Save rebalance state for future runs
            try:
                import pickle
                with open('rebalance_tracker_state.pkl', 'wb') as f:
                    pickle.dump(rebalance_tracker, f)
                logger.info("Saved rebalance tracker state for future runs")
            except Exception as e:
                logger.warning(f"Failed to save rebalance tracker state: {str(e)}")
        
        return allocation
    except Exception as e:
        logger.error(f"Error generating allocation: {str(e)}", exc_info=True)
        return None


def create_visualizations_pipeline(predictor, allocation_df, args):
    try:
        if args.show_attention:
            logger.info("Generating attention visualizations...")
            attention_weights = predictor.get_attention_weights(top_n=5)
            os.makedirs('plots', exist_ok=True)
            for ticker, weights in attention_weights.items():
                if hasattr(predictor, 'returns') and predictor.returns is not None and ticker in predictor.returns.columns:
                    returns_data_for_ticker = predictor.returns[ticker]
                else:
                    returns_data_for_ticker = None # Or load it specifically if needed
                create_prediction_plots( # From visualization_utils
                    ticker=ticker, 
                    weights=weights,
                    returns=returns_data_for_ticker,
                    interactive=args.interactive_plots, # Use the renamed arg
                    save_path='plots'
                )
        
        if allocation_df is not None and not allocation_df.empty:
            logger.info("Generating allocation visualizations...")
            # These are now assumed to be in allocation_utils
            visualize_optimized_allocation(allocation_df) 
            
            #  Store the sunburst figure
            sunburst_fig = None
            if args.interactive_plots: # Only show sunburst if interactive
                logger.info("Generating risk sunburst visualization...")
                sunburst_fig = plot_risk_sunburst(allocation_df)
                
                # keeping a reference to the figure 
                # in a global variable prevents it from being garbage collected
                import builtins
                if not hasattr(builtins, 'stored_figures'):
                    builtins.stored_figures = []
                builtins.stored_figures.append(sunburst_fig)
                
        return True
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
        return False


def run_benchmark(predictor, args):
    """Run performance benchmark for trend prediction models."""
    try:
        logger.info("Running performance benchmark...")
        benchmark_results = predictor.benchmark_performance(sample_size=10)
        
        if benchmark_results:
            # Save benchmark results
            import json
            with open('benchmark_results.json', 'w') as f:
                # Convert numpy values to Python primitives
                result_dict = {}
                for k, v in benchmark_results.items():
                    if k == 'aggregate_stats':
                        result_dict[k] = {}
                        for stat_k, stat_v in v.items():
                            result_dict[k][stat_k] = {
                                'mean': float(stat_v['mean']),
                                'min': float(stat_v['min']),
                                'max': float(stat_v['max']),
                                'std': float(stat_v['std'])
                            }
                    else:
                        result_dict[k] = {sub_k: float(sub_v) for sub_k, sub_v in v.items()}
                
                json.dump(result_dict, f, indent=2)
            
            logger.info("Benchmark results saved to benchmark_results.json")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_dashboard(args):
    """Run the ETF dashboard application."""
    try:
        from etf_dashboard import dashboard
        
        logger.info("Starting ETF Dashboard application...")
        build_models = args.train or args.force_rebuild
        dashboard(build_models=build_models)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
def run_etf_comparison(predictor, allocation_df, args):
    """
    Run ETF performance comparison and ensure visualizations persist.
    """
    logger.info("Running ETF performance comparison...")
    try:
        # from visualization_utils import create_alpha_vantage_etf_comparison, display_benchmark_metrics # Already imported
        metrics_comp, returns_comp = create_alpha_vantage_etf_comparison(
            predictor, allocation_df=allocation_df, 
            lookback_period=args.comparison_period, save_path='plots/etf_comparison'
        )
        
        if metrics_comp is not None and not metrics_comp.empty:
            # This is the modified function that will create persistent visualizations
            display_benchmark_metrics({'Strategy vs Benchmarks': metrics_comp})
            
            # Save results for future reference
            try:
                import pickle
                with open('etf_comparison_results.pkl', 'wb') as f:
                    pickle.dump((metrics_comp, returns_comp), f)
                logger.info("ETF comparison results saved.")
            except Exception as e:
                logger.warning(f"Failed to save ETF comparison results: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error in ETF comparison: {str(e)}", exc_info=True)
        return False

def main():
    args = parse_arguments()
    updated_model_params, updated_cuda_settings, updated_portfolio_settings = update_config_from_args(args)
    assets_dict = get_assets_dict(args)

    logger.info("=" * 80)
    logger.info("Starting Managed Futures ETF Strategy")
    logger.info(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    if torch.cuda.is_available() and updated_cuda_settings.get('enable_cuda'):
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}, Version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
    else:
        logger.info("CUDA not available or disabled. Using CPU.")
    logger.info(f"Effective Portfolio Settings: {updated_portfolio_settings}")
    logger.info(f"Effective Model Parameters: {updated_model_params}")
    logger.info(f"Effective CUDA Settings: {updated_cuda_settings}")
    logger.info(f"Assets Configuration: {assets_dict}")

     # preprocess data if training is requested
    if args.train:
        logger.info("Preprocessing data with enhanced quality controls...")
        enhanced_returns = preprocess_data_for_training(args)
        
        if enhanced_returns is not None:
            logger.info(f"Successfully preprocessed data for {len(enhanced_returns.columns)} assets")
        else:
            logger.warning("Data preprocessing failed or returned no data.")


    if args.dashboard:
        logger.info("Dashboard execution requested (ensure etf_dashboard.py is set up).")
        # run_dashboard(args) 
        return 0 # Placeholder for dashboard run

    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key and not args.load_model:
        logger.error("ALPHA_VANTAGE_API_KEY not set. Required for data fetching unless loading models. Exiting.")
        return 1
    
    try:
        from currency_patch import apply_currency_patches 
        apply_currency_patches(AlphaVantageClient)
        logger.info("Applied currency handling patches if defined.")
    except ImportError:
        logger.info("currency_patch.py not found, skipping.")
    except Exception as e:
        logger.warning(f"Failed to apply currency patches: {str(e)}")

    logger.info("Initializing CudaOptimizedTrendPredictor...")
    predictor = CudaOptimizedTrendPredictor(
        api_key=api_key,
        assets_dict=assets_dict,
        model_params=updated_model_params,
        cuda_settings=updated_cuda_settings
    )

    # Try to load previous rebalance state
    previous_rebalance_state = None
    if os.path.exists('rebalance_tracker_state.pkl'):
        try:
            import pickle
            with open('rebalance_tracker_state.pkl', 'rb') as f:
                previous_rebalance_state = pickle.load(f)
            logger.info("Loaded previous rebalance tracker state")
        except Exception as e:
            logger.warning(f"Failed to load previous rebalance tracker state: {str(e)}")

    # Initialize rebalance tracker
    if RebalanceTracker is not None:
        if previous_rebalance_state is not None and isinstance(previous_rebalance_state, RebalanceTracker):
            # Use previous state but update configuration
            rebalance_tracker = previous_rebalance_state
            rebalance_tracker.config = updated_portfolio_settings.get('rebalancing', {})
            logger.info("Using previous rebalance tracker state with updated configuration")
        else:
            # Create new tracker
            rebalance_tracker = RebalanceTracker(
                updated_portfolio_settings.get('rebalancing', {})
            )
        logger.info("Initialized rebalance tracker for hybrid rebalancing strategy")
    else:
        rebalance_tracker = None
        logger.warning("RebalanceTracker not available. Rebalancing checks will be skipped.")

    # If we have enhanced data and training was requested, use it
    if args.train and 'enhanced_returns' in locals() and enhanced_returns is not None:
        predictor.returns = enhanced_returns
    
    
    try:
        from advanced_features import generate_features_for_asset 
        
        original_load_technical_indicators_method = predictor.load_technical_indicators
        
        def enhanced_load_indicators_replacement_for_main(self_predictor_instance, use_cache=True, cache_file='cache/technical_indicators.pkl'):
            logger.info("Using main's ENHANCED load_technical_indicators with advanced_features fallback.")
            
            # Call original/base method first
            # The original method in CudaOptimizedTrendPredictor itself handles caching and fetching.
            self_predictor_instance.technical_indicators = original_load_technical_indicators_method(use_cache=use_cache, cache_file=cache_file)
            
            # Fallback logic using advanced_features
            missing_assets_for_features = []
            for asset_cls_iter, tickers_list_for_cls_iter in self_predictor_instance.assets_dict.items():
                for ticker_for_cls_iter in tickers_list_for_cls_iter:
                    # Check if indicators are truly missing or empty for the ticker
                    if ticker_for_cls_iter not in self_predictor_instance.technical_indicators or \
                       not self_predictor_instance.technical_indicators.get(ticker_for_cls_iter):
                        missing_assets_for_features.append((ticker_for_cls_iter, asset_cls_iter))
            
            if missing_assets_for_features:
                logger.info(f"Advanced features fallback for {len(missing_assets_for_features)} assets.")
                if self_predictor_instance.returns is None or self_predictor_instance.returns.empty:
                     logger.warning("Predictor returns data not loaded. Cannot generate advanced features. Load returns data first.")
                else:
                    for ticker_adv_main, asset_cls_adv_main in missing_assets_for_features:
                        if ticker_adv_main in self_predictor_instance.returns.columns:
                            # Use 'close' column if available, otherwise the series itself
                            # This assumes 'returns' DataFrame might contain 'close' prices for assets,
                            # or that generate_features_for_asset can handle raw returns series.
                            price_data_for_adv_feat = self_predictor_instance.returns[ticker_adv_main]
                            if isinstance(self_predictor_instance.returns, pd.DataFrame) and 'close' in self_predictor_instance.returns[ticker_adv_main].columns:
                                price_data_for_adv_feat = self_predictor_instance.returns[ticker_adv_main]['close']
                            
                            # Ensure it's a Series
                            if isinstance(price_data_for_adv_feat, pd.DataFrame):
                                price_data_for_adv_feat = price_data_for_adv_feat.iloc[:,0]


                            returns_for_adv_feat = price_data_for_adv_feat.pct_change().fillna(0)
                            
                            adv_features_df_main = generate_features_for_asset(price_data_for_adv_feat, asset_cls_adv_main, returns=returns_for_adv_feat)

                            if not adv_features_df_main.empty:
                                if ticker_adv_main not in self_predictor_instance.technical_indicators:
                                    self_predictor_instance.technical_indicators[ticker_adv_main] = {}
                                # Simplified mapping - real mapping depends on advanced_features output
                                if 'mean_20' in adv_features_df_main.columns:
                                    self_predictor_instance.technical_indicators[ticker_adv_main]['SMA_20_adv'] = pd.DataFrame({'SMA': adv_features_df_main['mean_20']})
                                if 'rsi_14' in adv_features_df_main.columns: # Check for common RSI name
                                     self_predictor_instance.technical_indicators[ticker_adv_main]['RSI_14_adv'] = pd.DataFrame({'RSI': adv_features_df_main['rsi_14']})
                                logger.info(f"Stored advanced features for {ticker_adv_main}.")
                        else:
                            logger.warning(f"No returns/price data for {ticker_adv_main} to generate advanced features.")
            return self_predictor_instance.technical_indicators

        predictor.load_technical_indicators = enhanced_load_indicators_replacement_for_main.__get__(predictor, CudaOptimizedTrendPredictor)
        logger.info("Patched load_technical_indicators in predictor instance with advanced features fallback.")
    except ImportError:
        logger.warning("advanced_features.py or its dependencies not found. Using standard features.")
    except Exception as e:
        logger.error(f"Failed to initialize advanced feature engineering integration: {str(e)}", exc_info=True)

    results = {}
    allocation_df = None 

    if args.load_model or args.predict or args.allocate:
        logger.info("Loading pre-trained models...")
        predictor.load_trend_models()

    if args.train:
        results['training'] = train_models_pipeline(predictor, args)
        if not results['training']: return 1 # Exit if training fails and was requested

    if args.predict or args.allocate:
        if not predict_trends_pipeline(predictor):
             logger.error("Prediction pipeline failed. Halting allocation/visualization.")
             return 1 

    # Apply timeframe diversification if enabled
    if args.allocate and args.enable_timeframe_diversification:
        try:
            logger.info("Applying timeframe diversification...")
            from timeframe_diversification import TimeframeDiversifiedPredictor # This should import the refactored version
            
            
            # For simplicity, using the defaults from the refactored TimeframeDiversifiedPredictor class:
            timeframe_keys_for_diversification = ['5d', '20d', '60d'] 
            

            # Pass the main predictor instance
            diversified_predictor_instance = TimeframeDiversifiedPredictor( # Renamed variable for clarity
                base_predictor=predictor, 
                timeframes=timeframe_keys_for_diversification 
                # Optionally pass regime_weights_override=current_portfolio_settings.get('custom_regime_weights')
            )
            
            # This now returns a Series of re-weighted signals (effectively new 'trend_predictions')
            new_trend_predictions = diversified_predictor_instance.predict_with_timeframe_diversification()
            
            if new_trend_predictions is not None and not new_trend_predictions.empty:
                logger.info("Timeframe diversification produced new trend predictions. Updating base predictor.")
                # Update the base predictor's signals
                # These will become the new base for economic/news adjustments
                predictor.trend_predictions = new_trend_predictions 
                predictor.trend_direction = pd.Series(np.sign(new_trend_predictions.reindex(predictor.trend_predictions.index).fillna(0)), index=new_trend_predictions.index).fillna(0).astype(int) # ensure same index and handle NaNs
                predictor.trend_strength = new_trend_predictions.abs().reindex(predictor.trend_predictions.index).fillna(0) # ensure same index and handle NaNs
                
                #  Re-apply economic and news adjustments on these new timeframe-diversified signals
                logger.info("Re-applying economic and news adjustments after timeframe diversification...")
                if hasattr(predictor, 'adjust_for_economic_uncertainty'):
                    # These methods should internally use predictor.trend_direction/strength
                    # and update predictor.adjusted_direction/strength then predictor.final_direction/strength
                    predictor.adjust_for_economic_uncertainty() 
                if hasattr(predictor, 'adjust_for_news_sentiment'):
                    predictor.adjust_for_news_sentiment()     
                
                logger.info("Successfully applied timeframe diversification and re-adjusted signals.")
            else:
                logger.warning("Timeframe diversification did not yield combined predictions. Using original predictions from base predictor.")
        
        except ImportError:
            logger.warning("Timeframe diversification module not available or error during import.")
        except Exception as e:
            logger.error(f"Error applying timeframe diversification: {str(e)}", exc_info=True)

    if args.allocate:
        logger.info("--- Generating Final Portfolio Allocation ---")
        # Use the updated_portfolio_settings which reflects command-line args or defaults
        allocation_df = generate_allocation_pipeline(
            predictor, 
            updated_portfolio_settings, 
            use_enhanced_allocation=True,
            rebalance_tracker=rebalance_tracker if 'rebalance_tracker' in locals() else None
        )
        results['allocation'] = allocation_df is not None and not allocation_df.empty
        
        if results['allocation']:
            logger.info("Allocation generated. Comparing with a 'standard/original' configuration for reference.")
            # For comparison, simulate an "original" setting, e.g., with lower commodity target
            original_settings_compare = updated_portfolio_settings.copy()
            original_settings_compare['asset_class_risk_allocation'] = original_settings_compare.get('asset_class_risk_allocation', {}).copy() # ensure deep copy
            original_settings_compare['asset_class_risk_allocation']['Commodities'] = 0.05 # Example original target
            original_settings_compare['commodity_target_leverage'] = 0.05

            original_allocation_compare = generate_allocation_pipeline(predictor, original_settings_compare, use_enhanced_allocation=False) # Call standard for this

            if original_allocation_compare is not None and not original_allocation_compare.empty:
                try:
                    compare_portfolio_allocations(original_allocation_compare, allocation_df) 
                    if 'Commodity_Sector' in allocation_df.columns: # Check if sector info is present
                        plot_commodity_sector_allocation(allocation_df) 
                except Exception as e_viz_comp:
                    logger.error(f"Error during allocation comparison visualization: {e_viz_comp}", exc_info=True)
            
            # Optional: Return-focused optimization (if portfolio_optimizer.py is robust)
            try:
                from portfolio_optimizer import optimize_portfolio_with_return_target #, visualize_optimized_portfolio
                logger.info("Attempting return-focused optimization on the enhanced allocation...")
                if 'Position' in allocation_df.columns and hasattr(predictor, 'returns') and predictor.returns is not None:
                    # Align returns and positions
                    common_idx = allocation_df.index.intersection(predictor.returns.columns)
                    pos_series_for_opt = allocation_df.loc[common_idx, 'Position']
                    returns_for_opt = predictor.returns[common_idx]

                    if not pos_series_for_opt.empty and not returns_for_opt.empty:
                        optimized_pos_series = optimize_portfolio_with_return_target(
                            allocations=pos_series_for_opt, returns_data=returns_for_opt,
                            risk_target=updated_portfolio_settings['risk_target'], return_target=0.15, # Example target
                            max_leverage=updated_portfolio_settings['max_leverage'],
                            max_position=updated_portfolio_settings.get('max_allocation_per_asset', 0.25)
                        )
                        if optimized_pos_series is not None and not optimized_pos_series.empty:
                            logger.info("Return-focused optimization successful. Updating allocation_df.")
                            allocation_df['Position_Optimized_Return'] = optimized_pos_series.reindex(allocation_df.index) # Add as new column or replace 'Position'
                            # visualize_optimized_portfolio(allocation_df['Position'], optimized_pos_series, predictor.returns) # Visualize the change
                            allocation_df.to_csv('final_return_optimized_allocation.csv')
                        else: logger.warning("Return-focused optimization did not yield results.")
                    else: logger.warning("Not enough common data for return-focused optimization.")
                else: logger.warning("Missing data for return-focused optimization.")
            except ImportError: logger.info("portfolio_optimizer.py not found. Skipping return optimization.")
            except Exception as e_ret_opt: logger.error(f"Error in return optimization: {e_ret_opt}", exc_info=True)

        # ETF Comparison
        if args.compare_etfs and allocation_df is not None and not allocation_df.empty:
            run_etf_comparison(predictor, allocation_df, args)

    if args.visualize and allocation_df is not None and not allocation_df.empty : # Ensure allocation_df is valid
        results['visualization'] = create_visualizations_pipeline(predictor, allocation_df, args)
    
    if args.benchmark:
        results['benchmark'] = run_benchmark(predictor, args) # Assuming run_benchmark is defined

    logger.info("=" * 80)
    logger.info("ETF Strategy Processing Summary:")
    for step, res_status_val in results.items(): 
        status_str_val = "SUCCESS" if res_status_val else ("SKIPPED" if res_status_val is None else "FAILED")
        logger.info(f"- {step.upper()}: {status_str_val}")
    logger.info("=" * 80)
    
    return 0

def keep_plots_visible():
    """
    Keep all plots visible until the user manually closes them or terminates the program.
    Call this at the end of main() before exiting.
    """
    import matplotlib.pyplot as plt
    import builtins
    
    if hasattr(builtins, 'stored_figures') and builtins.stored_figures:
        print(f"\nKeeping {len(builtins.stored_figures)} plots visible. Press Ctrl+C to exit when done viewing.")
        try:
            # This will block until all figures are closed
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nExiting due to user request.")
        finally:
            # Clean up
            plt.close('all')

if __name__ == "__main__":
    os.makedirs('plots/etf_comparison', exist_ok=True) 
    main_result = main()
    keep_plots_visible()  
    sys.exit(main_result)