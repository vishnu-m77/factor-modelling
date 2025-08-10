#!/usr/bin/env python3
"""
Linear Factor Calculator
Implements classic alpha factors: value, quality, momentum, size, volatility
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class LinearFactorCalculator:
    """Calculates classic linear alpha factors"""
    
    def __init__(self, price_data: Dict, fundamental_data: Dict):
        """
        Initialize factor calculator
        
        Args:
            price_data: Dictionary of price data by ticker
            fundamental_data: Dictionary of fundamental data by ticker
        """
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.factors = {}
    
    def calculate_momentum_factors(self):
        """Calculate momentum-based factors"""
        print("Calculating momentum factors...")
        
        momentum_factors = {}
        
        for ticker, price_data in self.price_data.items():
            if price_data.empty:
                continue
                
            factors = pd.DataFrame(index=price_data.index)
            
            # Price momentum (various lookback periods)
            for period in [5, 20, 60, 252]:
                factors[f'Momentum_{period}d'] = price_data['Adj Close'].pct_change(period)
            
            # Return momentum
            factors['Return_Momentum_5d'] = price_data['Returns'].rolling(5).mean()
            factors['Return_Momentum_20d'] = price_data['Returns'].rolling(20).mean()
            
            # Volatility-adjusted momentum
            factors['Sharpe_Momentum'] = factors['Momentum_20d'] / price_data['Volatility']
            
            momentum_factors[ticker] = factors
        
        return momentum_factors
    
    def calculate_value_factors(self):
        """Calculate value-based factors"""
        print("Calculating value factors...")
        
        value_factors = {}
        
        for ticker, fundamental_data in self.fundamental_data.items():
            if fundamental_data.empty:
                continue
                
            factors = pd.DataFrame(index=fundamental_data.index)
            
            # Price ratio factors
            if 'Price_Ratio' in fundamental_data.columns:
                factors['Price_Ratio'] = fundamental_data['Price_Ratio']
                factors['Price_Ratio_Rank'] = fundamental_data['Price_Ratio'].rolling(252).rank(pct=True)
            
            # Price momentum as value indicator
            if 'Price_Momentum' in fundamental_data.columns:
                factors['Price_Momentum_Value'] = -fundamental_data['Price_Momentum']  # Negative for value
            
            # Volatility-adjusted value
            if 'Volatility_Ratio' in fundamental_data.columns:
                factors['Volatility_Value'] = -fundamental_data['Volatility_Ratio']  # Negative for value
            
            value_factors[ticker] = factors
        
        return value_factors
    
    def calculate_quality_factors(self):
        """Calculate quality-based factors"""
        print("Calculating quality factors...")
        
        quality_factors = {}
        
        for ticker, fundamental_data in self.fundamental_data.items():
            if fundamental_data.empty:
                continue
                
            factors = pd.DataFrame(index=fundamental_data.index)
            
            # Quality proxy (inverse volatility)
            if 'Quality_Proxy' in fundamental_data.columns:
                factors['Quality_Score'] = fundamental_data['Quality_Proxy']
                factors['Quality_Rank'] = fundamental_data['Quality_Proxy'].rolling(252).rank(pct=True)
            
            # Volatility stability
            if 'Volatility_Ratio' in fundamental_data.columns:
                factors['Volatility_Stability'] = 1 / (1 + fundamental_data['Volatility_Ratio'])
            
            # Volume stability
            if 'Volume_Ratio' in fundamental_data.columns:
                factors['Volume_Stability'] = 1 / (1 + fundamental_data['Volume_Ratio'])
            
            quality_factors[ticker] = factors
        
        return quality_factors
    
    def calculate_size_factors(self):
        """Calculate size-based factors"""
        print("Calculating size factors...")
        
        size_factors = {}
        
        for ticker, fundamental_data in self.fundamental_data.items():
            if fundamental_data.empty:
                continue
                
            factors = pd.DataFrame(index=fundamental_data.index)
            
            # Size proxy
            if 'Size_Proxy' in fundamental_data.columns:
                factors['Size_Score'] = fundamental_data['Size_Proxy']
                factors['Size_Rank'] = fundamental_data['Size_Proxy'].rolling(252).rank(pct=True)
            
            # Size momentum
            if 'Size_Proxy' in fundamental_data.columns:
                factors['Size_Momentum'] = fundamental_data['Size_Proxy'].diff(20)
            
            size_factors[ticker] = factors
        
        return size_factors
    
    def calculate_volatility_factors(self):
        """Calculate volatility-based factors"""
        print("Calculating volatility factors...")
        
        volatility_factors = {}
        
        for ticker, price_data in self.price_data.items():
            if price_data.empty:
                continue
                
            factors = pd.DataFrame(index=price_data.index)
            
            # Volatility levels
            factors['Volatility_Level'] = price_data['Volatility']
            factors['Volatility_Rank'] = price_data['Volatility'].rolling(252).rank(pct=True)
            
            # Volatility momentum
            factors['Volatility_Momentum'] = price_data['Volatility'].diff(20)
            
            # Volatility ratio
            factors['Volatility_Ratio'] = price_data['Volatility'] / price_data['Volatility'].rolling(252).mean()
            
            # Volatility stability
            factors['Volatility_Stability'] = 1 / price_data['Volatility'].rolling(20).std()
            
            volatility_factors[ticker] = factors
        
        return volatility_factors
    
    def calculate_liquidity_factors(self):
        """Calculate liquidity-based factors"""
        print("Calculating liquidity factors...")
        
        liquidity_factors = {}
        
        for ticker, price_data in self.price_data.items():
            if price_data.empty:
                continue
                
            factors = pd.DataFrame(index=price_data.index)
            
            # Volume-based liquidity
            if 'Volume' in price_data.columns:
                factors['Volume_Liquidity'] = price_data['Volume']
                factors['Volume_Liquidity_Rank'] = price_data['Volume'].rolling(252).rank(pct=True)
                
                # Volume momentum
                factors['Volume_Momentum'] = price_data['Volume'].pct_change(20)
            
            # Price impact (inverse of liquidity)
            factors['Price_Impact'] = price_data['Returns'].abs() / (price_data['Volume'] + 1e-8)
            
            liquidity_factors[ticker] = factors
        
        return liquidity_factors
    
    def combine_factors(self, factor_dicts: List[Dict]) -> pd.DataFrame:
        """Combine all factor dictionaries into a single DataFrame"""
        print("Combining all factors...")
        
        all_factors = {}
        
        # Get common dates across all tickers
        common_dates = None
        for factor_dict in factor_dicts:
            for ticker, factors in factor_dict.items():
                if common_dates is None:
                    common_dates = set(factors.index)
                else:
                    common_dates = common_dates.intersection(set(factors.index))
        
        if common_dates is None:
            raise ValueError("No common dates found across factors")
        
        common_dates = sorted(list(common_dates))
        print(f"  Common dates: {len(common_dates)} observations")
        
        # Combine factors with ticker prefix
        for factor_dict in factor_dicts:
            for ticker, factors in factor_dict.items():
                if ticker in self.price_data:  # Only include tickers with price data
                    aligned_factors = factors.loc[common_dates]
                    for col in aligned_factors.columns:
                        all_factors[f"{ticker}_{col}"] = aligned_factors[col]
        
        combined_df = pd.DataFrame(all_factors, index=common_dates)
        print(f"  Combined factors: {combined_df.shape[1]} total factors")
        
        return combined_df
    
    def normalize_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize factors using z-score method"""
        print("Normalizing factors...")
        
        # Remove any remaining NaN values
        factors_df = factors_df.fillna(method='ffill').fillna(method='bfill')
        
        # Z-score normalization
        normalized = (factors_df - factors_df.rolling(252, min_periods=50).mean()) / factors_df.rolling(252, min_periods=50).std()
        
        # Clip extreme values
        normalized = normalized.clip(-3, 3)
        
        print(f"  Normalized factors: {normalized.shape[1]} factors")
        return normalized
    
    def calculate_all_factors(self) -> pd.DataFrame:
        """Calculate all factor types and return combined DataFrame"""
        print("Calculating all linear factors...")
        
        # Calculate each factor type
        momentum_factors = self.calculate_momentum_factors()
        value_factors = self.calculate_value_factors()
        quality_factors = self.calculate_quality_factors()
        size_factors = self.calculate_size_factors()
        volatility_factors = self.calculate_volatility_factors()
        liquidity_factors = self.calculate_liquidity_factors()
        
        # Combine all factors
        all_factors = self.combine_factors([
            momentum_factors, value_factors, quality_factors,
            size_factors, volatility_factors, liquidity_factors
        ])
        
        # Normalize factors
        normalized_factors = self.normalize_factors(all_factors)
        
        print(f"Linear factor calculation completed: {normalized_factors.shape[1]} factors")
        return normalized_factors

def main():
    """Test the linear factor calculator"""
    # This would typically be called from the main pipeline
    print("Linear Factor Calculator - Test Mode")
    print("This module is designed to be called from the main pipeline")

if __name__ == "__main__":
    main()
