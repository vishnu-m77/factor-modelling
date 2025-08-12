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
        if 'Adj Close' not in price_data.columns and 'Close' not in price_data.columns:
            return pd.Series()
        
        # Handle new yfinance column format
        if isinstance(price_data.columns, pd.MultiIndex):
            price_col = ('Close', ticker) if ('Close', ticker) in price_data.columns else ('Adj Close', ticker)
        else:
            price_col = 'Adj Close' if 'Adj Close' in price_data.columns else 'Close'
        
        # Calculate forward returns
        forward_returns = price_data[price_col].shift(-period) / price_data[price_col] - 1
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
    
    def calculate_ic_statistics(self, ic_values: List[float]) -> Dict:
        """Calculate comprehensive IC statistics"""
        if not ic_values or all(np.isnan(ic_values)):
            return {
                'mean': np.nan, 'std': np.nan, 't_stat': np.nan,
                'hit_rate': np.nan, 'ic_ir': np.nan, 'ic_ir_t': np.nan
            }
        
        # Remove NaN values
        valid_ics = [ic for ic in ic_values if not np.isnan(ic)]
        if not valid_ics:
            return {
                'mean': np.nan, 'std': np.nan, 't_stat': np.nan,
                'hit_rate': np.nan, 'ic_ir': np.nan, 'ic_ir_t': np.nan
            }
        
        ic_array = np.array(valid_ics)
        n = len(ic_array)
        
        # Basic statistics
        mean_ic = np.mean(ic_array)
        std_ic = np.std(ic_array, ddof=1)
        
        # T-statistic for IC significance
        t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
        
        # Hit rate (percentage of positive ICs)
        hit_rate = np.mean(ic_array > 0)
        
        # Information Ratio (IC mean / IC std)
        ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
        
        # IR t-statistic
        ic_ir_t = ic_ir * np.sqrt(n) if not np.isnan(ic_ir) else np.nan
        
        return {
            'mean': mean_ic,
            'std': std_ic,
            't_stat': t_stat,
            'hit_rate': hit_rate,
            'ic_ir': ic_ir,
            'ic_ir_t': ic_ir_t
        }
    
    def evaluate_single_factor(self, factor_name: str) -> Dict:
        """Evaluate a single factor across all forward periods"""
        factor_values = self.factors[factor_name]
        
        ic_by_period = {}
        ic_stats_by_period = {}
        
        for period in self.forward_periods:
            # Calculate IC for each ticker
            ic_values = []
            for ticker in self.price_data.keys():
                forward_returns = self.calculate_forward_returns(ticker, period)
                ic = self.calculate_ic(factor_values, forward_returns)
                if not np.isnan(ic):
                    ic_values.append(ic)
            
            # Calculate IC statistics
            ic_stats = self.calculate_ic_statistics(ic_values)
            
            # Store results
            ic_by_period[f'IC_{period}d'] = ic_stats['mean']
            ic_stats_by_period[f'IC_{period}d_Stats'] = ic_stats
        
        # Calculate overall IC score (average across periods)
        valid_ics = [ic for ic in ic_by_period.values() if not np.isnan(ic)]
        if valid_ics:
            overall_ic = np.mean(valid_ics)
            overall_ic_std = np.std(valid_ics, ddof=1)
            overall_ic_t = overall_ic / (overall_ic_std / np.sqrt(len(valid_ics))) if overall_ic_std > 0 else np.nan
        else:
            overall_ic = np.nan
            overall_ic_t = np.nan
        
        # Combine results
        result = {**ic_by_period}
        result['Overall_IC'] = overall_ic
        result['Overall_IC_t'] = overall_ic_t
        
        # Add detailed statistics for the most predictive period
        best_period = max(ic_by_period.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
        if not np.isnan(best_period[1]):
            best_stats = ic_stats_by_period[f'{best_period[0]}_Stats']
            result['Best_Period'] = best_period[0]
            result['Best_IC'] = best_period[1]
            result['Best_IC_t'] = best_stats['t_stat']
            result['Best_IC_IR'] = best_stats['ic_ir']
            result['Best_IC_HitRate'] = best_stats['hit_rate']
        
        return result
    
    def evaluate_all_factors(self) -> pd.DataFrame:
        """Evaluate all factors and return IC results"""
        print("Evaluating factors using IC analysis...")
        
        results = []
        total_factors = len(self.factors.columns)
        
        for i, factor_name in enumerate(self.factors.columns):
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{total_factors}")
            
            try:
                factor_result = self.evaluate_single_factor(factor_name)
                factor_result['Factor'] = factor_name
                results.append(factor_result)
            except Exception as e:
                print(f"    Error evaluating {factor_name}: {str(e)}")
                continue
        
        # Convert to DataFrame
        ic_df = pd.DataFrame(results).set_index('Factor')
        
        # Sort by overall IC significance (absolute value)
        ic_df['IC_Significance'] = ic_df['Overall_IC'].abs()
        ic_df = ic_df.sort_values('IC_Significance', ascending=False)
        
        self.ic_results = ic_df
        print(f"IC evaluation completed: {len(ic_df)} factors evaluated")
        
        return ic_df
    
    def get_top_factors(self, n_top: int = 20, min_ic_threshold: float = 0.005, 
                       min_t_stat: float = 1.0, min_hit_rate: float = 0.45) -> pd.DataFrame:
        """Get top factors based on enhanced IC screening criteria"""
        if self.ic_results is None or self.ic_results.empty:
            raise ValueError("Must run evaluate_all_factors() first")
        
        print(f"Selecting top {n_top} factors with enhanced screening...")
        
        # Apply multiple screening criteria
        valid_factors = self.ic_results.copy()
        
        # 1. IC threshold (more lenient)
        ic_mask = valid_factors['Overall_IC'].abs() >= min_ic_threshold
        print(f"  Factors meeting IC threshold {min_ic_threshold}: {ic_mask.sum()}")
        
        # 2. T-statistic threshold
        t_mask = valid_factors['Overall_IC_t'].abs() >= min_t_stat
        print(f"  Factors meeting t-stat threshold {min_t_stat}: {t_mask.sum()}")
        
        # 3. Hit rate threshold (if available)
        hit_mask = pd.Series(True, index=valid_factors.index)
        if 'Best_IC_HitRate' in valid_factors.columns:
            hit_mask = valid_factors['Best_IC_HitRate'] >= min_hit_rate
            print(f"  Factors meeting hit rate threshold {min_hit_rate}: {hit_mask.sum()}")
        
        # Combine all masks
        final_mask = ic_mask & t_mask & hit_mask
        valid_factors = valid_factors[final_mask]
        
        if valid_factors.empty:
            print(f"  No factors meet all screening criteria")
            # Fallback to just IC threshold
            valid_factors = self.ic_results[self.ic_results['Overall_IC'].abs() >= min_ic_threshold * 0.5]
            if valid_factors.empty:
                return pd.DataFrame()
            print(f"  Fallback: {len(valid_factors)} factors meet relaxed IC threshold")
        
        # Sort by IC significance
        sorted_factors = valid_factors.sort_values('IC_Significance', ascending=False)
        
        # Select top factors
        top_factors = sorted_factors.head(n_top)
        
        # Get the actual factor values
        top_factor_names = top_factors.index.tolist()
        top_factor_values = self.factors[top_factor_names]
        
        self.top_factors = top_factor_values
        
        print(f"  Selected {len(top_factors)} top factors")
        print(f"  IC range: {top_factors['Overall_IC'].min():.4f} to {top_factors['Overall_IC'].max():.4f}")
        print(f"  T-stat range: {top_factors['Overall_IC_t'].min():.2f} to {top_factors['Overall_IC_t'].max():.2f}")
        
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

    def analyze_ic_decay(self, factor_name: str = None) -> pd.DataFrame:
        """Analyze IC decay across different forward periods"""
        if self.ic_results is None or self.ic_results.empty:
            raise ValueError("Must run evaluate_all_factors() first")
        
        print("Analyzing IC decay patterns...")
        
        if factor_name is None:
            # Analyze top factors
            top_factors = self.ic_results.head(10).index.tolist()
        else:
            top_factors = [factor_name]
        
        decay_data = []
        for factor in top_factors:
            factor_decay = {'Factor': factor}
            
            # Extract IC values for each period
            for period in self.forward_periods:
                ic_col = f'IC_{period}d'
                if ic_col in self.ic_results.columns:
                    factor_decay[f'{period}d'] = self.ic_results.loc[factor, ic_col]
                else:
                    factor_decay[f'{period}d'] = np.nan
            
            decay_data.append(factor_decay)
        
        decay_df = pd.DataFrame(decay_data).set_index('Factor')
        
        # Calculate decay metrics
        for factor in decay_df.index:
            ic_values = decay_df.loc[factor].dropna()
            if len(ic_values) > 1:
                # IC persistence (correlation between consecutive periods)
                periods = [int(col.replace('d', '')) for col in ic_values.index]
                ic_array = ic_values.values
                
                if len(ic_array) > 1:
                    # Calculate decay rate (change in IC per day)
                    decay_rate = (ic_array[-1] - ic_array[0]) / (periods[-1] - periods[0])
                    decay_df.loc[factor, 'Decay_Rate'] = decay_rate
                    
                    # IC stability (std of IC across periods)
                    decay_df.loc[factor, 'IC_Stability'] = np.std(ic_array)
                    
                    # Best period
                    best_period_idx = np.argmax(np.abs(ic_array))
                    decay_df.loc[factor, 'Best_Period'] = periods[best_period_idx]
                    decay_df.loc[factor, 'Best_IC'] = ic_array[best_period_idx]
        
        print(f"  Analyzed IC decay for {len(decay_df)} factors")
        return decay_df
    
    def get_factor_quality_score(self, factor_name: str) -> Dict:
        """Calculate comprehensive quality score for a factor"""
        if self.ic_results is None or factor_name not in self.ic_results.index:
            raise ValueError(f"Factor {factor_name} not found in IC results")
        
        factor_data = self.ic_results.loc[factor_name]
        
        # Base quality score components
        quality_score = 0.0
        max_score = 100.0
        
        # 1. IC magnitude (30 points)
        ic_magnitude = abs(factor_data.get('Overall_IC', 0))
        ic_score = min(30, ic_magnitude * 100)
        quality_score += ic_score
        
        # 2. IC significance (25 points)
        ic_t_stat = abs(factor_data.get('Overall_IC_t', 0))
        t_score = min(25, ic_t_stat * 2.5)
        quality_score += t_score
        
        # 3. Hit rate (20 points)
        hit_rate = factor_data.get('Best_IC_HitRate', 0.5)
        if not np.isnan(hit_rate):
            hit_score = min(20, (hit_rate - 0.5) * 40)
            quality_score += hit_score
        
        # 4. IC stability (15 points)
        ic_stability = factor_data.get('IC_Stability', 1.0)
        if not np.isnan(ic_stability):
            stability_score = max(0, 15 - ic_stability * 15)
            quality_score += stability_score
        
        # 5. Information ratio (10 points)
        ic_ir = abs(factor_data.get('Best_IC_IR', 0))
        ir_score = min(10, ic_ir * 10)
        quality_score += ir_score
        
        # Quality grade
        if quality_score >= 80:
            grade = 'A'
        elif quality_score >= 60:
            grade = 'B'
        elif quality_score >= 40:
            grade = 'C'
        elif quality_score >= 20:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'total_score': quality_score,
            'grade': grade,
            'components': {
                'ic_magnitude': ic_score,
                'ic_significance': t_score,
                'hit_rate': hit_score if 'hit_score' in locals() else 0,
                'ic_stability': stability_score if 'stability_score' in locals() else 0,
                'information_ratio': ir_score
            }
        }
    
    def generate_enhanced_report(self) -> Dict:
        """Generate comprehensive enhanced factor evaluation report"""
        if self.ic_results is None:
            raise ValueError("Must run evaluate_all_factors() first")
        
        print("Generating enhanced factor evaluation report...")
        
        # Basic statistics
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
            'quality_distribution': {
                'excellent': 0,  # A grade
                'good': 0,       # B grade
                'fair': 0,       # C grade
                'poor': 0,       # D grade
                'failing': 0     # F grade
            }
        }
        
        # Analyze top factors
        top_factors = self.ic_results.head(20)
        report['top_factors'] = {}
        
        for factor in top_factors.index:
            quality = self.get_factor_quality_score(factor)
            report['top_factors'][factor] = quality
            
            # Update quality distribution
            grade = quality['grade']
            if grade == 'A':
                report['quality_distribution']['excellent'] += 1
            elif grade == 'B':
                report['quality_distribution']['good'] += 1
            elif grade == 'C':
                report['quality_distribution']['fair'] += 1
            elif grade == 'D':
                report['quality_distribution']['poor'] += 1
            else:
                report['quality_distribution']['failing'] += 1
        
        # IC decay analysis
        try:
            decay_analysis = self.analyze_ic_decay()
            report['ic_decay_analysis'] = decay_analysis.to_dict('index')
        except Exception as e:
            report['ic_decay_analysis'] = f"Error: {str(e)}"
        
        print("  Enhanced report generated successfully")
        return report
    
    def save_ic_analysis(self, output_dir: str = 'results'):
        """Save detailed IC analysis results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.ic_results is None:
            print("No IC results to save. Run evaluate_all_factors() first.")
            return
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main IC results
        ic_file = f"{output_dir}/enhanced_ic_results_{timestamp}.csv"
        self.ic_results.to_csv(ic_file)
        print(f"Enhanced IC results saved to: {ic_file}")
        
        # Save IC decay analysis
        try:
            decay_analysis = self.analyze_ic_decay()
            decay_file = f"{output_dir}/ic_decay_analysis_{timestamp}.csv"
            decay_analysis.to_csv(decay_file)
            print(f"IC decay analysis saved to: {decay_file}")
        except Exception as e:
            print(f"Could not save IC decay analysis: {str(e)}")
        
        # Save enhanced report
        try:
            enhanced_report = self.generate_enhanced_report()
            import json
            report_file = f"{output_dir}/enhanced_factor_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(enhanced_report, f, indent=2, default=str)
            print(f"Enhanced factor report saved to: {report_file}")
        except Exception as e:
            print(f"Could not save enhanced report: {str(e)}")

def main():
    """Test the factor IC evaluator"""
    print("Factor IC Evaluator - Test Mode")
    print("This module is designed to be called from the main pipeline")

if __name__ == "__main__":
    main() 