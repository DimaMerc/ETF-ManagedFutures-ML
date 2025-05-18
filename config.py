# config.py - Modified for compatibility with your existing alpha_vantage_client

"""Configuration settings for the enhanced managed futures ETF strategy."""

# Asset classes and tickers - using the direct commodity symbols your client already supports
ASSETS = {
    'Commodities': [
        # Direct commodity symbols
        'BRENT',
        'WTI',
        'NATURAL_GAS',
        'COPPER',
        'ALUMINUM',
        'WHEAT',
        'CORN',
    ],
    'Currencies': [
        # Single currency codes
        'EUR',  # EUR/USD is implied
        'GBP',  # GBP/USD is implied
        'JPY',  # USD/JPY is implied
        'CAD',  # USD/CAD is implied
        'AUD',  # AUD/USD is implied
        'CHF',  # USD/CHF is implied
        'NZD',  # NZD/USD is implied
    ],
    'Bonds': ['TLT', 'IEF', 'SHY', 'GOVT', 'BND'],
    'Equities': ['SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'VGK']

}

MODEL_PARAMS = {
    'sequence_length': 20,  # Maintained
    'hidden_sizes': [64, 32, 16],  # Reverted to original sizes
    'num_layers': 2,  # Reverted to original number of layers
    'dropout': 0.2,  # Reverted to original dropout rate
    'epochs': 50,  # Maintained
    'batch_size': 64,  # Reverted to original batch size
    'learning_rate': 0.001,  # Maintained
    'weight_decay': 1e-5,  # Maintained
    'attention_heads': 2,  # Reverted to original number of attention heads
    'early_stopping_patience': 10  # Maintained
}

# CUDA settings remain unchanged
CUDA_SETTINGS = {
    'enable_cuda': True,
    'enable_mixed_precision': True,
    'enable_cuda_graphs': True,
    'batch_size': 64,  # Reverted to match MODEL_PARAMS
    'set_float32_precision': 'high',
    'enable_cudnn_benchmark': True,
    'use_torch_compile': True,
    'use_tensor_cores': True,
    'pin_memory': True,
    'enable_channels_last': False,
    'compile_mode': 'reduce-overhead',
    'enable_dynamic_shapes': False,
}

# Enhanced portfolio settings with increased commodity allocation
PORTFOLIO_SETTINGS = {
    'max_allocation_per_asset': 0.20,      # Maintained at 20%
    'risk_target': 0.15,                   # Maintained at 15%
    'rebalance_threshold': 0.05,           # Maintained at 5%
    'max_leverage': 1.8,                   # Maintained at 2.0x
    'asset_class_risk_allocation': {       # Modified to increase commodities
        'Commodities': 0.20,               # Increased from 0.05 to 0.20
        'Currencies': 0.25,                # Reduced from 0.30 to 0.25
        'Bonds': 0.25,                     # Maintained at 0.25
        'Equities': 0.30                   # Reduced from 0.40 to 0.30
    },

    'commodity_target_leverage': 0.20, # Explicit 20% GROSS LEVERAGE target for commodities
    'currency_target_leverage': 0.15, # Explicit 15% GROSS LEVERAGE for currencies

    'initial_signal_multiplier': 100.0,       # Multiplier for raw signals before filtering (TUNE THIS)
    'base_expected_return_per_risk_unit': 0.5, # Assumed Sharpe for a perfect signal (TUNE THIS)
    'max_expected_signal_strength_for_er': 0.5, # Max raw signal strength used for ER calc normalization (TUNE THIS)
    'final_leverage_upscale_cap_factor': 1.5,
    'min_position_size': 0.01,

    'min_active_signals_before_seeding': 5,
    'seed_position_size_if_few_signals' : 0.01, 

    'min_signal_strength': 0.05,          # Maintained
    'rebalance_frequency': 'weekly',       # Changed from monthly to weekly for better commodity tracking
    'trade_execution_lag': 1,              # Maintained
    'use_economic_regime_overlay': True,   # Maintained
    
    # New sub-sector allocations for commodities - using your existing system
    'commodity_sector_weights': {
        'Energy': 0.40,                    # 8% of total portfolio (40% of 20%)
        'Metal': 0.30,                     # 6% of total portfolio (30% of 20%)
        'Agriculture': 0.30,               # 6% of total portfolio (30% of 20%)

     # directional bias parameters
    'allow_directional_bias': False,
    'max_net_exposure': 0.15,              # Allow up to 40% net market exposure
    'trend_following_bias': 0.20           # Allow stronger trend following
    },
    
       # Minimum allocation percentages by asset class
    'min_allocation_percentage': {
        'Commodities': 0.10,    # At least 15% in commodities 
        'Bonds': 0.10,          # At least 15% in bonds
        'Equities': 0.10,        # At least 15% in equities
        'Currencies': 0.05
    },
    
    # Lower minimum position size for commodities
    'min_position_size_commodities': 0.01,  # 0.25% min position for commodities
    
    # Minimum sector percentage (of total commodity allocation)
    'min_sector_percentage': 0.15,  # Each sector at least 15% of commodity allocation

    'min_asset_classes_present': 4, # Ensure at least these many desired classes have *some* position
    'forced_min_diversification_position_size': 0.01, # e.g. 1% if forcing a class

    # This should match your CommodityData._get_commodity_type method
    'commodity_sector_mapping': {
        'BRENT': 'Energy',
        'WTI': 'Energy',
        'NATURAL_GAS': 'Energy',
        'COPPER': 'Metal',
        'ALUMINUM': 'Metal',
        'WHEAT': 'Agriculture',
        'CORN': 'Agriculture',
    },

    # Hybrid rebalancing settings
    'rebalancing': {
        'enable_quarterly_rebalance': True,
        'enable_signal_driven_rebalance': True,
        'signal_change_threshold': 0.15,
        'min_days_between_rebalances': 14,
        'force_initial_allocation': True
    }
}

# AlphaVantage settings - already handled by your client
ALPHA_VANTAGE_SETTINGS = {
    'cache_days': 1,
    'lookback_days': 730,
    'technical_indicators': [
        {'name': 'SMA', 'periods': [20, 50, 200]},
        {'name': 'EMA', 'periods': [20, 50, 200]},
        {'name': 'RSI', 'periods': [14]},
        {'name': 'MACD', 'periods': [12, 26, 9]},
        {'name': 'BBANDS', 'periods': [20]}
    ],
    'economic_indicators': [
        'REAL_GDP',
        'TREASURY_YIELD',
        'CPI',
        'INFLATION',
        'RETAIL_SALES',
        'UNEMPLOYMENT',
        'INDUSTRIAL_PRODUCTION',
    ],
    'use_alternate_data': True,
    'enable_sentiment_analysis': True
}