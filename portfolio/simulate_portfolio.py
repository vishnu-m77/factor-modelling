#!/usr/bin/env python3
"""
Portfolio Simulation Module
Implements long-short portfolio construction from top factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PortfolioSimulator:
    """Simulates long-short portfolios using top factors"""
    
    def __init__(self, factors: pd.DataFrame, price_data: Dict, 
                 rebalance_freq: int = 21, transaction_cost: float = 0.001):
        """
        Initialize portfolio simulator
        
        Args:
            factors: DataFrame of top factors
            price_data: Dictionary of price data by ticker
            rebalance_freq: Rebalancing frequency in days
            transaction_cost: Transaction cost as fraction
        """
        self.factors = factors
        self.price_data = price_data
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.portfolio_returns = None
        self.positions = None
        
    def calculate_factor_weights(self, factor_values: pd.Series) -> pd.Series:
        """Calculate portfolio weights based on factor values"""
        # Z-score normalization
        normalized = (factor_values - factor_values.rolling(252, min_periods=50).mean()) / factor_values.rolling(252, min_periods=50).std()
        
        # Clip extreme values
        normalized = normalized.clip(-3, 3)
        
        # Convert to weights (long-short)
        weights = normalized / normalized.abs().sum()
        
        return weights
    
    def calculate_portfolio_weights(self, date: pd.Timestamp) -> Dict[str, float]:
        """Calculate portfolio weights for a given date"""
        if date not in self.factors.index:
            return {}
        
        # Get factor values for the date
        factor_values = self.factors.loc[date]
        
        # Calculate weights for each factor
        all_weights = {}
        
        for factor_name in factor_values.index:
            if pd.isna(factor_values[factor_name]):
                continue
                
            # Extract ticker from factor name
            ticker = factor_name.split('_')[0]
            
            if ticker in self.price_data:
                # Calculate factor weight
                factor_weight = self.calculate_factor_weights(self.factors[factor_name])
                
                if date in factor_weight.index:
                    weight = factor_weight.loc[date]
                    if not pd.isna(weight):
                        all_weights[ticker] = all_weights.get(ticker, 0) + weight
        
        # Normalize weights to sum to 0 (long-short)
        if all_weights:
            total_weight = sum(all_weights.values())
            if total_weight != 0:
                for ticker in all_weights:
                    all_weights[ticker] /= total_weight
        
        return all_weights
    
    def calculate_portfolio_returns(self, weights: Dict[str, float], date: pd.Timestamp) -> float:
        """Calculate portfolio returns for given weights and date"""
        if not weights:
            return 0.0
        
        portfolio_return = 0.0
        
        for ticker, weight in weights.items():
            if ticker in self.price_data:
                price_data = self.price_data[ticker]
                
                if date in price_data.index and 'Returns' in price_data.columns:
                    ticker_return = price_data.loc[date, 'Returns']
                    if not pd.isna(ticker_return):
                        portfolio_return += weight * ticker_return
        
        return portfolio_return
    
    def apply_transaction_costs(self, old_weights: Dict[str, float], 
                              new_weights: Dict[str, float]) -> float:
        """Calculate transaction costs for rebalancing"""
        if not old_weights and not new_weights:
            return 0.0
        
        # Calculate turnover
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        total_turnover = 0.0
        
        for ticker in all_tickers:
            old_weight = old_weights.get(ticker, 0.0)
            new_weight = new_weights.get(ticker, 0.0)
            turnover = abs(new_weight - old_weight)
            total_turnover += turnover
        
        # Apply transaction costs
        transaction_cost = total_turnover * self.transaction_cost
        return transaction_cost
    
    def simulate_portfolio(self) -> Dict:
        """Simulate portfolio performance"""
        print("Simulating portfolio performance...")
        
        # Get common dates
        common_dates = self.factors.index
        if not common_dates.empty:
            common_dates = common_dates.sort_index()
        
        # Initialize tracking variables
        portfolio_values = [1.0]  # Start with $1
        portfolio_returns = []
        positions_history = []
        transaction_costs = []
        
        current_weights = {}
        current_date = None
        
        for i, date in enumerate(common_dates):
            # Calculate new weights (rebalance every rebalance_freq days)
            if i % self.rebalance_freq == 0 or current_weights == {}:
                new_weights = self.calculate_portfolio_weights(date)
                
                # Calculate transaction costs
                if current_weights:
                    cost = self.apply_transaction_costs(current_weights, new_weights)
                    transaction_costs.append(cost)
                else:
                    transaction_costs.append(0.0)
                
                current_weights = new_weights
                current_date = date
            
            # Calculate portfolio return
            if current_weights:
                portfolio_return = self.calculate_portfolio_returns(current_weights, date)
                
                # Subtract transaction costs
                if transaction_costs:
                    portfolio_return -= transaction_costs[-1]
                
                portfolio_returns.append(portfolio_return)
                
                # Update portfolio value
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
                
                # Record positions
                positions_history.append(current_weights.copy())
            else:
                portfolio_returns.append(0.0)
                portfolio_values.append(portfolio_values[-1])
                positions_history.append({})
        
        # Create results
        results = {
            'dates': common_dates,
            'portfolio_values': portfolio_values[1:],  # Remove initial $1
            'portfolio_returns': portfolio_returns,
            'positions': positions_history,
            'transaction_costs': transaction_costs
        }
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(results)
        results.update(performance_metrics)
        
        print("Portfolio simulation completed")
        return results
    
    def calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate portfolio performance metrics"""
        print("Calculating performance metrics...")
        
        returns = pd.Series(results['portfolio_returns'])
        values = pd.Series(results['portfolio_values'])
        
        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1 if len(values) > 1 else 0
        annualized_return = ((1 + total_return) ** (252 / len(returns))) - 1 if len(returns) > 1 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0
        
        # VaR and CVaR (95% confidence)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'total_days': len(returns)
        }
        
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Win Rate: {win_rate:.2%}")
        
        return metrics
    
    def simulate(self) -> Dict:
        """Main method to run portfolio simulation"""
        return self.simulate_portfolio()
    
    def get_position_summary(self, results: Dict) -> pd.DataFrame:
        """Get summary of portfolio positions over time"""
        if 'positions' not in results:
            return pd.DataFrame()
        
        # Convert positions to DataFrame
        positions_df = pd.DataFrame(results['positions'], index=results['dates'])
        
        # Calculate position statistics
        position_stats = {}
        for ticker in positions_df.columns:
            ticker_positions = positions_df[ticker].dropna()
            if len(ticker_positions) > 0:
                position_stats[ticker] = {
                    'avg_weight': ticker_positions.mean(),
                    'max_weight': ticker_positions.max(),
                    'min_weight': ticker_positions.min(),
                    'weight_std': ticker_positions.std(),
                    'appearance_freq': (ticker_positions != 0).mean()
                }
        
        return pd.DataFrame(position_stats).T

def main():
    """Test the portfolio simulator"""
    print("Portfolio Simulator - Test Mode")
    print("This module is designed to be called from the main pipeline")

if __name__ == "__main__":
    main()
