# Alpha Factor Research: ML-Based Factor Evaluation Pipeline

This repository contains a modular Python pipeline to research and evaluate the predictive power of various linear and nonlinear alpha factors using free financial data. It aims to reflect realistic constraints in quant research — no premium data feeds, no real-time execution, and runs entirely locally.

---

## Project Goals

- Research modern alpha factors using free fundamental and price data
- Implement machine learning models to extract nonlinear factor structure
- Measure factor effectiveness using forward Information Coefficient (IC)
- Simulate basic long-short strategies using top-ranked alpha signals
- Modularize the codebase for easy extension with new models, features, and datasets

---

## Project Structure

```bash
.
├── data/
│   └── download.py              # Download and preprocess price & fundamental data
│
├── factors/
│   ├── linear_factors.py        # Classic alpha factors (e.g., value, quality, momentum)
│   └── nonlinear_factors.py     # ML-based features (e.g., autoencoders, nonlinear PCA)
│
├── models/
│   ├── encoder_models.py        # Autoencoder and nonlinear dimensionality models
│   └── factor_ic_eval.py        # Factor ranking and forward IC analysis
│
├── portfolio/
│   └── simulate_portfolio.py    # Construct long-short portfolios from top factors
│
├── utils/
│   └── helpers.py               # Utilities for returns, plotting, evaluation, etc.
│
├── config.py                    # Configuration parameters and settings
├── requirements.txt             # Python dependencies and versions
├── main_pipeline.py             # End-to-end script: from data to factor IC results
├── results/                     # Output directory for analysis results and plots
├── cache/                       # Data caching directory for downloaded financial data
└── README.md
```

---

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
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

### Factor Calculation
- **Linear Factors**: Value, quality, momentum, size, volatility
- **Nonlinear Factors**: Autoencoder-based features, nonlinear PCA
- Factor normalization and outlier detection

### Factor Evaluation
- Forward Information Coefficient (IC) analysis
- Factor ranking and selection
- Correlation analysis and filtering

### Portfolio Simulation
- Long-short portfolio construction
- Performance metrics calculation
- Risk management and position sizing

## Output Files

The pipeline generates comprehensive output files:

### Core Results
- ic_results_[timestamp].csv: Basic IC analysis results
- top_factors_[timestamp].csv: Selected top factors
- portfolio_results_[timestamp].json: Portfolio simulation results
- all_factors_[timestamp].csv: Complete factor dataset
- complete_results_[timestamp].pkl: Pickled complete results

### IC Analysis
- enhanced_ic_results_[timestamp].csv: Detailed IC statistics with t-stats, hit rates, and IR
- ic_decay_analysis_[timestamp].csv: IC decay patterns across forward periods
- enhanced_factor_report_[timestamp].json: Comprehensive factor evaluation report

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
```

---

## Configuration

Key parameters can be modified in the respective modules:
- Data sources and date ranges
- Factor calculation parameters
- Model hyperparameters
- Portfolio constraints
- IC Screening Parameters
---

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- yfinance
- torch (for autoencoder models)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Update documentation
5. Submit a pull request

---

## License

This project is for research purposes. Please ensure compliance with data usage terms and trading regulations.

---

**Note**: This is a research framework. Results should be validated and backtested before any real trading applications.
