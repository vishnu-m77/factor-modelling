# Alpha Factor Research: ML-Based Factor Evaluation Pipeline

This repository contains a modular Python pipeline to research and evaluate the predictive power of various linear and nonlinear alpha factors using free financial data. It aims to reflect realistic constraints in quant research — no premium data feeds, no real-time execution, and runs entirely locally.

---

## Project Goals

- Research modern alpha factors using free fundamental and price data
- Implement machine learning models to extract nonlinear factor structure
- Measure factor effectiveness using forward Information Coefficient (IC)
- Simulate basic long-short strategies using top-ranked alpha signals
- Modularize the codebase for easy extension with new models, features, and datasets
- **NEW**: Prevent overfitting through comprehensive cross-validation and out-of-sample testing

---

## Project Structure

```bash
.
├── data/
│   └── download.py              # Download and preprocess price & fundamental data
│
├── factors/
│   ├── linear_factors.py        # Classic alpha factors (e.g., value, quality, momentum)
│   ├── nonlinear_factors.py     # ML-based features (e.g., autoencoders, nonlinear PCA)
│   └── enhanced_factor_processing.py  # Advanced factor processing with robust normalization
│
├── models/
│   ├── encoder_models.py        # Autoencoder and nonlinear dimensionality models
│   ├── factor_ic_eval.py        # Factor ranking and forward IC analysis
│   ├── factor_quality_control.py # Advanced factor quality control and validation
│   ├── cross_validation.py      # Time series CV, walk-forward analysis, out-of-sample testing
│   └── validation_functions.py  # Validation functions for cross-validation framework
│
├── portfolio/
│   └── simulate_portfolio.py    # Construct long-short portfolios from top factors
│
├── utils/
│   └── helpers.py               # Utilities for data handling, results saving, and reporting
│
├── config.py                    # Configuration parameters and settings
├── requirements.txt             # Python dependencies and versions
├── main_pipeline.py             # End-to-end script: from data to factor IC results
├── results/                     # Output directory for analysis results and plots
├── cache/                       # Data caching directory for downloaded financial data
├── logs/                        # Pipeline execution logs
├── existing_scripts/            # Reference implementations and research notes
└── README.md
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vishnu-m77/factor-modelling
cd factor-modelling

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python main_pipeline.py
```

---

## Features

### Data Pipeline
- Downloads free financial data using yfinance
- Preprocesses price and fundamental data
- Implements caching to avoid redundant downloads
- Handles missing data with robust imputation strategies

### Advanced Factor Processing
- **Robust Normalization**: IQR-based outlier handling with rolling z-score normalization
- **Multi-Strategy Imputation**: Forward/backward fill, median, and zero-fill strategies
- **Ensemble Factor Creation**: PCA and ICA factors for enhanced diversification
- **Quality-Based Selection**: IC-based factor selection with correlation filtering

### Advanced Quality Control
- **Multi-criteria Factor Validation**: Comprehensive factor quality assessment using IC, data quality, stability, and outlier detection
- **Outlier Detection**: Isolation Forest-based outlier identification for robust factor analysis
- **Stability Analysis**: Rolling volatility-based factor stability scoring
- **Quality Scoring**: Weighted quality scores combining IC performance, data quality, stability, and outlier ratios
- **Automated Filtering**: Intelligent factor selection based on configurable quality thresholds

### Factor Calculation
- **Linear Factors**: Value, quality, momentum, size, volatility, liquidity (156 total factors)
- **Nonlinear Factors**: Autoencoder-based features, nonlinear PCA, ensemble combinations
- **Enhanced Processing**: Robust normalization, outlier detection, and quality filtering

### Factor Evaluation
- Forward Information Coefficient (IC) analysis across multiple time horizons
- Factor ranking and selection with enhanced screening criteria
- Correlation analysis and filtering to reduce redundancy
- Multi-criteria factor validation including outlier detection, stability analysis, and comprehensive quality scoring

### Cross-Validation & Out-of-Sample Testing
- **Time Series Cross-Validation**: Prevents overfitting with temporal data splitting
- **Walk-Forward Analysis**: Rolling window validation for realistic testing
- **Out-of-Sample Testing**: Temporal data splits to ensure generalization
- **Factor Stability Analysis**: Rolling window IC analysis for consistency assessment
- **Overfitting Detection**: Comprehensive analysis of train/test performance degradation

### Portfolio Simulation
- Long-short portfolio construction with realistic constraints
- Performance metrics calculation (Sharpe ratio, drawdown, win rate, VaR)
- Risk management and position sizing
- Transaction cost modeling

## Output Files

The pipeline generates comprehensive output files:

### Core Results
- `ic_results_[timestamp].csv`: Basic IC analysis results
- `top_factors_[timestamp].csv`: Selected top factors
- `portfolio_results_[timestamp].json`: Portfolio simulation results
- `all_factors_[timestamp].csv`: Complete factor dataset
- `complete_results_[timestamp].pkl`: Pickled complete results containing ALL data

### Cross-Validation Results
- **Cross-validation results**: Stored within the pickled file
- **Walk-forward analysis**: Stored within the pickled file  
- **Out-of-sample testing**: Stored within the pickled file
- **Factor stability analysis**: Stored within the pickled file

### Quality Control Results
- **Quality control results**: Stored within the pickled file
- **Quality reports**: Stored within the pickled file
- **Processing results**: Stored within the pickled file

### IC Analysis
- `enhanced_ic_results_[timestamp].csv`: Detailed IC statistics with t-stats, hit rates, and IR
- `ic_decay_analysis_[timestamp].csv`: IC decay patterns across forward periods
- `enhanced_factor_report_[timestamp].json`: Comprehensive factor evaluation report

---

## Usage Examples

### Run Complete Pipeline
```bash
python main_pipeline.py
```

### Individual Components
```bash
# Download data
python data/download.py

# Calculate linear factors
python factors/linear_factors.py

# Extract nonlinear factors
python factors/nonlinear_factors.py

# Evaluate factors
python models/factor_ic_eval.py

# Simulate portfolio
python portfolio/simulate_portfolio.py

# Run cross-validation
python models/cross_validation.py

# Run quality control
python models/factor_quality_control.py
```

---

## Configuration

Key parameters can be modified in the respective modules:
- Data sources and date ranges
- Factor calculation parameters
- Model hyperparameters
- Portfolio constraints
- IC Screening Parameters
- Cross-validation parameters (folds, window sizes, test ratios)
- Quality control thresholds (IC, stability, outlier detection)

---

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- yfinance
- torch
- scikit-learn

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Update documentation
5. Submit a pull request

