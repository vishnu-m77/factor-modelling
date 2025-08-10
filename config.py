"""
Configuration file for the Factor Modeling Project
Contains all parameters, data sources, and model settings
"""

import os
from datetime import datetime

# Data Configuration
TICKERS = ["MTUM", "VLUE", "QUAL", "USMV"]  # Factor ETFs
BENCHMARK = "SPY"  # Market benchmark
MACRO_INDICATORS = ["CPIAUCSL", "UNRATE"]  # Optional macro indicators

# Date Ranges
TRAIN_START = "2015-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2023-12-31"

# Feature Windows
FEATURE_WINDOWS = {
    "momentum": [21, 63, 126],
    "volatility": [21, 63],
    "rolling_sharpe": [63, 126],
    "technical": [14, 21, 63]
}

# Transaction Costs and Constraints
TRANSACTION_COST = 0.001  # 10 basis points
MAX_TURNOVER = 0.2  # 20% maximum turnover
MAX_POSITION_SIZE = 0.3  # 30% maximum position size

# Regime Classification
REGIME_WINDOWS = {
    "volatility": 21,
    "momentum": 63
}

# Model Parameters
MODEL_PARAMS = {
    "ppo": {
        "learning_rate": 3e-4,
        "total_timesteps": 20000,
        "batch_size": 64,
        "n_steps": 2048
    },
    "q_learning": {
        "alpha": 0.2,
        "gamma": 0.9,
        "rar": 0.5,
        "radr": 0.99,
        "dyna": 0
    }
}

# Factor Parameters
FACTOR_PARAMS = {
    "ic_lookback": 252,  # 1 year for IC calculation
    "min_ic_threshold": 0.02,
    "max_factor_correlation": 0.7
}

# Portfolio Parameters
PORTFOLIO_PARAMS = {
    "rebalance_frequency": "monthly",
    "risk_free_rate": 0.02,
    "target_volatility": 0.15
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data Storage
DATA_DIR = "data"
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Environment Variables (for API keys if needed in future)
API_KEYS = {
    "yfinance": None,  # yfinance is free
    "fred": None,  # FRED API is free with registration
    "quandl": None  # Some datasets are free
}

# Validation
def validate_config():
    """Validate configuration parameters"""
    assert TRANSACTION_COST >= 0, "Transaction cost must be non-negative"
    assert MAX_TURNOVER > 0 and MAX_TURNOVER <= 1, "Max turnover must be between 0 and 1"
    assert MAX_POSITION_SIZE > 0 and MAX_POSITION_SIZE <= 1, "Max position size must be between 0 and 1"
    assert len(TICKERS) > 0, "Must specify at least one ticker"
    
    # Validate date ranges
    train_start = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    train_end = datetime.strptime(TRAIN_END, "%Y-%m-%d")
    test_start = datetime.strptime(TEST_START, "%Y-%m-%d")
    test_end = datetime.strptime(TEST_END, "%Y-%m-%d")
    
    assert train_start < train_end, "Train start must be before train end"
    assert test_start < test_end, "Test start must be before test end"
    assert train_end <= test_start, "Train and test periods should not overlap"

# Run validation
if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed!") 