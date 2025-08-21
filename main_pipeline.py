#!/usr/bin/env python3
"""
Main Pipeline for Alpha Factor Research
Implements end-to-end pipeline from data download to factor IC results
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data.download import DataDownloader
from factors.linear_factors import LinearFactorCalculator
from factors.nonlinear_factors import NonlinearFactorCalculator
from models.factor_ic_eval import FactorICEvaluator
from portfolio.simulate_portfolio import PortfolioSimulator
from utils.helpers import setup_logging, save_results

def main():
    """Main execution pipeline"""
    print("Starting Alpha Factor Research Pipeline")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Pipeline started")
    
    try:
        # Step 1: Download and preprocess data
        print("\n1. Downloading financial data...")
        data_downloader = DataDownloader()
        price_data, fundamental_data = data_downloader.run()
        logger.info(f"Downloaded {len(price_data)} price observations and {len(fundamental_data)} fundamental observations")
        
        # Step 2: Calculate linear factors
        print("\n2. Calculating linear factors...")
        linear_calc = LinearFactorCalculator(price_data, fundamental_data)
        linear_factors = linear_calc.calculate_all_factors()
        logger.info(f"Calculated {len(linear_factors.columns)} linear factors")
        
        # Step 3: Extract nonlinear factors
        print("\n3. Extracting nonlinear factors...")
        nonlinear_calc = NonlinearFactorCalculator(linear_factors)
        nonlinear_factors = nonlinear_calc.extract_factors()
        logger.info(f"Extracted {len(nonlinear_factors.columns)} nonlinear factors")
        
        # Step 4: Combine all factors
        all_factors = pd.concat([linear_factors, nonlinear_factors], axis=1)
        logger.info(f"Total factors: {len(all_factors.columns)}")
        
        # Step 5: Evaluate factors using IC analysis
        print("\n4. Evaluating factors with IC analysis...")
        ic_evaluator = FactorICEvaluator(all_factors, price_data)
        ic_results = ic_evaluator.evaluate_all_factors()
        
        # Enhanced IC screening with multiple criteria
        top_factors = ic_evaluator.get_top_factors(
            n_top=20, 
            min_ic_threshold=0.005,  # More lenient IC threshold
            min_t_stat=1.0,          # Minimum t-statistic
            min_hit_rate=0.45        # Minimum hit rate
        )
        
        # Save enhanced IC analysis
        ic_evaluator.save_ic_analysis()
        
        logger.info(f"Top factors identified: {len(top_factors)}")
        
        # Generate and display enhanced report
        if len(top_factors) > 0:
            enhanced_report = ic_evaluator.generate_enhanced_report()
            print(f"\nEnhanced Factor Analysis:")
            print(f"  Total factors analyzed: {enhanced_report['summary']['total_factors']}")
            print(f"  Mean IC: {enhanced_report['ic_statistics']['mean_ic']:.4f}")
            print(f"  IC std: {enhanced_report['ic_statistics']['std_ic']:.4f}")
            print(f"  Quality distribution:")
            for grade, count in enhanced_report['quality_distribution'].items():
                if count > 0:
                    print(f"    {grade.capitalize()}: {count} factors")
        else:
            print("\nNo factors met the enhanced screening criteria.")
            print("Consider relaxing thresholds or improving factor quality.")
        
        # Step 6: Simulate portfolio
        print("\n5. Simulating portfolio...")
        # Use the actual factor values, not the IC statistics
        portfolio_sim = PortfolioSimulator(all_factors, price_data)
        portfolio_results = portfolio_sim.simulate()
        logger.info("Portfolio simulation completed")
        
        # Step 7: Save results
        print("\n6. Saving results...")
        results = {
            'ic_results': ic_results,
            'top_factors': top_factors,
            'portfolio_results': portfolio_results,
            'all_factors': all_factors
        }
        save_results(results)
        logger.info("Results saved successfully")
        
        # Step 8: Print summary
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Total factors analyzed: {len(all_factors.columns)}")
        print(f"Top factors selected: {len(top_factors)}")
        print(f"Portfolio Sharpe ratio: {portfolio_results.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"Portfolio total return: {portfolio_results.get('total_return', 'N/A'):.1%}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\nERROR: Pipeline failed - {str(e)}")
        raise
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
