#!/usr/bin/env python3
"""
Helper utilities for alpha factor research
"""

import logging
import os
import json
import pandas as pd
from datetime import datetime
import pickle

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_results(results, output_dir='results'):
    """Save pipeline results to files"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save IC results
    if 'ic_results' in results:
        ic_file = f"{output_dir}/ic_results_{timestamp}.csv"
        results['ic_results'].to_csv(ic_file)
        print(f"IC results saved to: {ic_file}")
    
    # Save top factors
    if 'top_factors' in results:
        factors_file = f"{output_dir}/top_factors_{timestamp}.csv"
        results['top_factors'].to_csv(factors_file)
        print(f"Top factors saved to: {factors_file}")
    
    # Save portfolio results
    if 'portfolio_results' in results:
        portfolio_file = f"{output_dir}/portfolio_results_{timestamp}.json"
        with open(portfolio_file, 'w') as f:
            json.dump(results['portfolio_results'], f, indent=2, default=str)
        print(f"Portfolio results saved to: {portfolio_file}")
    
    # Save all factors
    if 'all_factors' in results:
        all_factors_file = f"{output_dir}/all_factors_{timestamp}.csv"
        results['all_factors'].to_csv(all_factors_file)
        print(f"All factors saved to: {all_factors_file}")
    
    # Save complete results as pickle for later analysis
    pickle_file = f"{output_dir}/complete_results_{timestamp}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Complete results saved to: {pickle_file}")

def load_results(file_path):
    """Load saved results from file"""
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def create_summary_report(results, output_file='results/summary_report.txt'):
    """Create a text summary report of the results"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("ALPHA FACTOR RESEARCH SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Factor summary
        if 'all_factors' in results:
            f.write(f"Total factors analyzed: {len(results['all_factors'].columns)}\n")
            f.write(f"Time period: {results['all_factors'].index[0]} to {results['all_factors'].index[-1]}\n\n")
        
        # IC results summary
        if 'ic_results' in results:
            ic_data = results['ic_results']
            f.write("FACTOR IC ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average IC: {ic_data['IC'].mean():.4f}\n")
            f.write(f"IC t-stat: {ic_data['IC_tstat'].mean():.4f}\n")
            f.write(f"IC hit rate: {ic_data['IC_hit_rate'].mean():.4f}\n\n")
        
        # Top factors
        if 'top_factors' in results:
            f.write("TOP FACTORS SELECTED:\n")
            f.write("-" * 20 + "\n")
            for col in results['top_factors'].columns:
                f.write(f"- {col}\n")
            f.write("\n")
        
        # Portfolio results
        if 'portfolio_results' in results:
            portfolio = results['portfolio_results']
            f.write("PORTFOLIO PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total return: {portfolio.get('total_return', 'N/A'):.2%}\n")
            f.write(f"Sharpe ratio: {portfolio.get('sharpe_ratio', 'N/A'):.3f}\n")
            f.write(f"Max drawdown: {portfolio.get('max_drawdown', 'N/A'):.2%}\n")
            f.write(f"Volatility: {portfolio.get('volatility', 'N/A'):.2%}\n")
    
    print(f"Summary report saved to: {output_file}")
