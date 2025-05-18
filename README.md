# ETF-ManagedFutures-ML
1. Introduction and Objectives
1.1 Project Objectives
The primary objective of this research was to develop a systematic, machine learning-enhanced managed futures strategy that could:
1.	Generate positive absolute returns across diverse market environments
2.	Maintain competitive risk-adjusted performance metrics (Sharpe, Calmar ratios)
3.	Deliver effective portfolio diversification benefits
4.	Provide meaningful allocation to commodity markets (≥20% of portfolio)
5.	Capitalize on market trends across multiple asset classes and timeframes
1.2 Scope and Methodology
The project encompassed the full investment process from data acquisition to portfolio construction and performance analysis:
•	Historical price and returns data acquisition from financial APIs
•	Feature engineering and technical indicator generation
•	Machine learning model development using LSTM neural networks
•	Multi-asset trend signal generation and processing
•	Dynamic portfolio construction algorithms
•	Hybrid rebalancing framework implementation
•	Performance attribution and backtest analysis
2. Background on Managed Futures
2.1 Asset Class Context
Managed futures strategies typically involve systematic trading across multiple liquid futures markets, including commodities, currencies, fixed income, and equity index futures. These strategies aim to capture directional price movements (trends) across various market regimes while providing diversification benefits to traditional portfolios.
2.2 Key Strategy Characteristics
Traditional managed futures strategies are characterized by:
•	Trend-following algorithms that identify and capitalize on price momentum
•	Diversification across multiple asset classes and markets
•	Ability to take both long and short positions
•	Dynamic allocation and position sizing
•	Risk-managed exposure targeting specific volatility levels
Our implementation builds upon these foundations while incorporating advanced machine learning techniques, sophisticated rebalancing, and enhanced portfolio optimization.
3. System Architecture and Implementation
3.1 Software Architecture
The system was implemented using a modular Python architecture with the following key components:
├── main.py                   # Primary execution module
├── config.py                 # Configuration parameters
├── cuda_optimized_trend_predictor.py  # Core prediction engine
├── allocation_utils.py       # Portfolio construction algorithms
├── alpha_vantage_client_optimized.py  # Data acquisition
├── rebalancing.py            # Hybrid rebalancing framework
├── simple_lstm_model.py      # Neural network model definitions
├── timeframe_diversification.py  # Multi-timeframe signal processing
└── visualization_utils.py    # Performance analytics and visualization
3.2 Data Pipeline
Historical financial data was sourced from Alpha Vantage API, covering:
•	7 commodity markets (energy, metals, agriculture)
•	7 currency pairs
•	5 bond ETFs across the yield curve
•	7 equity indices (domestic, international, and emerging markets)
The data preprocessing pipeline included:
1.	Price to returns conversion
2.	Data quality verification and anomaly detection
3.	ETF proxy mapping for assets with limited history
4.	Feature engineering for technical indicators
5.	Time-series normalization and sequence preparation
4. Model Design and Signal Generation
4.1 Neural Network Architecture
The trend prediction component utilized Long Short-Term Memory (LSTM) neural networks with the following specifications:
•	Sequential input: 20-day price and indicator sequences
•	Hidden layer architecture: [64, 32, 16] with bidirectional processing
•	Dropout rate: 0.2 for regularization
•	Attention mechanism for focusing on relevant time steps
•	Training epochs: 50 with early stopping patience of 10
•	Batch size: 64 with mixed precision training
Model training incorporated cross-validation and regularization techniques to prevent overfitting on historical data patterns.
4.2 Signal Generation Process
The signal generation workflow consisted of:
1.	Technical feature extraction across multiple timeframes
2.	LSTM model prediction of trend direction and magnitude
3.	Economic regime overlay (expansion, slowdown, contraction, recovery)
4.	News sentiment adjustment for short-term signal modulation
5.	Timeframe diversification with regime-specific weightings
6.	Signal strength normalization and directional adjustment
4.3 CUDA Optimization
The prediction pipeline was accelerated using NVIDIA CUDA, enabling:
•	Parallel model training and inference
•	Mixed precision (FP16/FP32) calculations
•	Dynamic graph optimization
•	Tensor core utilization for matrix operations
•	Memory optimization for batch processing
5. Portfolio Construction Methodology
5.1 Asset Allocation Framework
The portfolio allocation algorithm was implemented with multiple objectives:
•	Target volatility scaling (15% annualized volatility)
•	Fixed asset class allocations (20% commodities, 20% bonds, 30% equities, 10% currencies)
•	Maximum leverage constraint (1.8x gross exposure)
•	Maximum position size limits (25% per asset)
•	Minimum diversification requirements across asset classes
5.2 Signal Processing and Position Sizing
Position sizes were determined through a multi-step process:
1.	Initial position sizing based on signal strength (magnitude × direction)
2.	Commodity sector balancing (energy, metals, agriculture)
3.	Trend alignment with known market directions
4.	Asset-class diversification enforcement
5.	Historical performance tilt based on asset class returns
6.	Risk scaling to target portfolio volatility
7.	Net exposure control within defined bounds (-15% to +15%)
8.	Benchmark-relative allocation component for stability
6. Rebalancing Strategy
6.1 Hybrid Rebalancing Framework
A novel hybrid rebalancing approach was implemented that combines:
•	Scheduled quarterly rebalancing for stability
•	Signal-driven rebalancing based on trend reversals
•	Economic regime shift triggers
•	Minimum time between rebalances (14 days)
•	Maximum rebalancing frequency bounds
6.2 Rebalancing Conditions
Rebalancing decisions were governed by:
1.	Direction reversals in major assets (SPY, TLT, BRENT)
2.	Significant signal magnitude changes (>15% threshold)
3.	Average signal changes across the portfolio (>7.5% threshold)
4.	Shifts in economic regime classification
5.	Quarterly calendar boundaries (January, April, July, October)
7. Performance Analysis
7.1 Backtest Results
The strategy was backtested over a complete market cycle (365 days) with the following results:
Metric	ML Managed Futures	S&P 500 ETF	Bond ETF	60/40 Portfolio	DBMF ETF
Annual Return	12.74%	21.10%	4.94%	14.80%	0.72%
Volatility	12.20%	17.92%	5.64%	11.25%	10.42%
Sharpe Ratio	1.04	1.18	0.88	1.32	0.07
Max Drawdown	9.43%	18.76%	4.82%	11.27%	15.60%
Calmar Ratio	1.35	1.13	1.02	1.31	0.05
7.2 Risk-Return Characteristics
The strategy demonstrated several notable characteristics:
1.	Superior drawdown protection compared to equity markets (9.43% vs 18.76%)
2.	Higher Calmar ratio than all benchmarks (1.35)
3.	Higher returns than fixed income and professional managed futures product
4.	Moderate correlation to traditional assets
5.	Step-function return profile reflecting the hybrid rebalancing approach
8. Technical Infrastructure
8.1 Hardware Specifications
The development and implementation utilized:
•	CPU: 24-core Intel(R) Core(TM) Ultra 9 275HX
•	GPU: NVIDIA RTX 5090 with 24GB VRAM
•	Memory: 64GB DDR5 RAM

8.2 Software Stack
The implementation leveraged the following software technologies:
•	Python 3.10 for core implementation
•	PyTorch 2.7 with CUDA 12.8 for neural network operations
•	Pandas/NumPy for data manipulation
•	Scikit-learn for preprocessing and evaluation
•	Matplotlib/Plotly for visualization
•	Alpha Vantage API for financial data acquisition
8.3 Optimization Techniques
Several optimization approaches were employed:
•	CUDA acceleration for model training and inference
•	Mixed precision arithmetic for faster computation
•	Dynamic batch sizing for optimal memory utilization
•	Caching strategy for frequently accessed data
•	Parallel processing for independent calculations
•	Tensor core utilization for matrix operations
9. Limitations and Future Work
9.1 Current Limitations
The implementation has several limitations that warrant consideration:
1.	Limited historical data for comprehensive out-of-sample testing
2.	Sensitivity to parameter choices in the allocation algorithm
3.	Step-like return profile may indicate suboptimal rebalancing frequency
4.	Lack of explicit transaction cost modeling
5.	Limited asset universe compared to commercial implementations
9.2 Future Enhancement Opportunities
Potential areas for future improvement include:
1.	Extended Data History: Incorporate longer historical data series for more robust training and testing
2.	Advanced Feature Engineering: Implement macroeconomic factors and alternative data sources
3.	Refined Rebalancing: Develop more sophisticated rebalance triggers based on performance metrics
4.	Dynamic Leverage: Implement regime-dependent leverage utilization
5.	Expanded Asset Universe: Include additional commodities, credit instruments, and alternative markets
6.	Transaction Cost Optimization: Incorporate execution modeling and cost minimization
7.	Reinforcement Learning: Explore RL frameworks for direct optimization of allocation decisions
10. Conclusion
This research successfully developed a machine learning-enhanced managed futures strategy that provides competitive returns with strong risk-adjusted performance metrics. The implementation demonstrates the viability of neural network approaches to financial trend prediction when combined with thoughtful portfolio construction and rebalancing methodologies.
The strategy's performance significantly outpaced professional managed futures products while offering better drawdown protection than traditional equity investments. The hybrid rebalancing approach provided an effective balance between portfolio stability and timely adaptation to changing market conditions.
While several limitations remain for future exploration, the framework established in this project provides a solid foundation for further refinement and potential commercial application of machine learning techniques in systematic investment strategies.
Acknowledgments
This research acknowledges the use of the Alpha Vantage API for financial data and NVIDIA's CUDA platform for accelerated computation. The project was conducted for academic purposes and is not intended as a commercial product offering.

