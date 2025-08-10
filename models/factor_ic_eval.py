#!/usr/bin/env python3
"""
Factor IC Evaluation Module
Implements factor ranking and forward Information Coefficient analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FactorICEvaluator:
    """Evaluates factors using Information Coefficient (IC) analysis"""
    
    def __init__(self, factors: pd.DataFrame, price_data: Dict, forward_periods: List[int] = [1, 5, 21]):
        """
        Initialize IC evaluator
        
        Args:
            factors: DataFrame of factors
            price_data: Dictionary of price data by ticker
            forward_periods: List of forward periods for IC calculation
        """
        self.factors = factors
        self.price_data = price_data
        self.forward_periods = forward_periods
        self.ic_results = {}
        self.top_factors = None
        
    def calculate_forward_returns(self, ticker: str, period: int) -> pd.Series:
        """Calculate forward returns for a given period"""
        if ticker not in self.price_data:
            return pd.Series()
        
        price_data = self.price_data[ticker]
        if 'Adj Close' not in price_data.columns:
            return pd.Series()
        
        # Calculate forward returns
        forward_returns = price_data['Adj Close'].shift(-period) / price_data['Adj Close'] - 1
        return forward_returns
    
    def calculate_ic(self, factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """Calculate Information Coefficient between factor and forward returns"""
        # Align data
        aligned_data = pd.concat([factor_values, forward_returns], axis=1).dropna()
        
        if len(aligned_data) < 50:  # Need sufficient data
            return np.nan
        
        # Calculate correlation
        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
        return correlation
    
    def evaluate_single_factor(self, factor_name: str) -> Dict:
        """Evaluate a single factor across all forward periods"""
        factor_values = self.factors[factor_name]
        
        ic_by_period = {}
        ic_rank_by_period = {}
        
        for period in self.forward_periods:
            # Calculate IC for each ticker
            ic_values = []
            for ticker in self.price_data.keys():
                forward_returns = self.calculate_forward_returns(ticker, period)
                ic = self.calculate_ic(factor_values, forward_returns)
                if not np.isnan(ic):
                    ic_values.append(ic)
            
            # Calculate average IC across tickers
            if ic_values:
                avg_ic = np.mean(ic_values)
                ic_by_period[f'IC_{period}d'] = avg_ic
                ic_rank_by_period[f'IC_{period}d_Rank'] = avg_ic
            else:
                ic_by_period[f'IC_{period}d'] = np.nan
                ic_rank_by_period[f'IC_{period}d_Rank'] = np.nan
        
        # Calculate overall IC score
        valid_ics = [ic for ic in ic_by_period.values() if not np.isnan(ic)]
        if valid_ics:
            overall_ic = np.mean(valid_ics)
            ic_rank_by_period['Overall_IC'] = overall_ic
        else:
            ic_rank_by_period['Overall_IC'] = np.nan
        
        return {**ic_by_period, **ic_rank_by_period}
    
    def evaluate_all_factors(self) -> pd.DataFrame:
        """Evaluate all factors and return IC results"""
        print("Evaluating factors using IC analysis...")
        
        results = []
        total_factors = len(self.factors.columns)
        
        for i, factor_name in enumerate(self.factors.columns):
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{total_factors}")
            
            factor_result = self.evaluate_single_factor(factor_name)
            factor_result['Factor_Name'] = factor_name
            results.append(factor_result)
        
        # Create results DataFrame
        ic_df = pd.DataFrame(results)
        ic_df = ic_df.set_index('Factor_Name')
        
        # Calculate rankings
        for col in ic_df.columns:
            if 'IC' in col and 'Rank' not in col:
                rank_col = f'{col}_Rank'
                if rank_col in ic_df.columns:
                    ic_df[rank_col] = ic_df[col].rank(ascending=False)
        
        self.ic_results = ic_df
        print(f"IC evaluation completed: {len(ic_df)} factors evaluated")
        
        return ic_df
    
    def get_top_factors(self, n_top: int = 20, min_ic_threshold: float = 0.01) -> pd.DataFrame:
        """Get top factors based on IC performance"""
        if self.ic_results is None or self.ic_results.empty:
            raise ValueError("Must run evaluate_all_factors() first")
        
        print(f"Selecting top {n_top} factors...")
        
        # Filter factors with minimum IC threshold
        valid_factors = self.ic_results[self.ic_results['Overall_IC'].abs() >= min_ic_threshold]
        
        if valid_factors.empty:
            print(f"  No factors meet IC threshold {min_ic_threshold}")
            return pd.DataFrame()
        
        # Sort by overall IC (absolute value)
        sorted_factors = valid_factors.sort_values('Overall_IC', key=abs, ascending=False)
        
        # Select top factors
        top_factors = sorted_factors.head(n_top)
        
        # Get the actual factor values
        top_factor_names = top_factors.index.tolist()
        top_factor_values = self.factors[top_factor_names]
        
        self.top_factors = top_factor_values
        
        print(f"  Selected {len(top_factors)} top factors")
        print(f"  IC range: {top_factors['Overall_IC'].min():.4f} to {top_factors['Overall_IC'].max():.4f}")
        
        return top_factors
    
    def analyze_factor_correlations(self, factors: pd.DataFrame = None) -> pd.DataFrame:
        """Analyze correlations between factors"""
        if factors is None:
            factors = self.factors
        
        print("Analyzing factor correlations...")
        
        # Calculate correlation matrix
        correlation_matrix = factors.corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'Factor1': correlation_matrix.columns[i],
                        'Factor2': correlation_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"  Found {len(high_corr_pairs)} high correlation pairs (|corr| > 0.8)")
            for pair in high_corr_pairs[:5]:  # Show first 5
                print(f"    {pair['Factor1']} <-> {pair['Factor2']}: {pair['Correlation']:.3f}")
        else:
            print("  No high correlations found")
        
        return correlation_matrix
    
    def generate_factor_report(self) -> Dict:
        """Generate comprehensive factor evaluation report"""
        if self.ic_results is None:
            raise ValueError("Must run evaluate_all_factors() first")
        
        print("Generating factor evaluation report...")
        
        report = {
            'summary': {
                'total_factors': len(self.ic_results),
                'forward_periods': self.forward_periods,
                'evaluation_date': pd.Timestamp.now()
            },
            'ic_statistics': {
                'mean_ic': self.ic_results['Overall_IC'].mean(),
                'std_ic': self.ic_results['Overall_IC'].std(),
                'min_ic': self.ic_results['Overall_IC'].min(),
                'max_ic': self.ic_results['Overall_IC'].max(),
                'positive_ic_count': (self.ic_results['Overall_IC'] > 0).sum(),
                'negative_ic_count': (self.ic_results['Overall_IC'] < 0).sum()
            },
            'top_factors': self.ic_results.head(10).to_dict('index') if len(self.ic_results) > 0 else {},
            'bottom_factors': self.ic_results.tail(10).to_dict('index') if len(self.ic_results) > 0 else {}
        }
        
        print("  Report generated successfully")
        return report

def main():
    """Test the factor IC evaluator"""
    print("Factor IC Evaluator - Test Mode")
    print("This module is designed to be called from the main pipeline")

if __name__ == "__main__":
    main() 