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
from factors.enhanced_factor_processing import EnhancedFactorProcessor
from models.factor_ic_eval import FactorICEvaluator
from models.factor_quality_control import FactorQualityController
from models.cross_validation import TimeSeriesCrossValidator, WalkForwardValidator, OutOfSampleTester
from models.validation_functions import validate_factor_model_comprehensive, validate_factor_stability, validate_factor_model_single_dataset
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
        raw_factors = pd.concat([linear_factors, nonlinear_factors], axis=1)
        logger.info(f"Total raw factors: {len(raw_factors.columns)}")
        
        # Step 4.5: Apply enhanced factor processing
        print("\n4.5. Applying enhanced factor processing...")
        enhanced_processor = EnhancedFactorProcessor(
            min_ic=0.005,  # Lower threshold to get more factors
            min_observations=50,
            max_missing_ratio=0.2,
            correlation_threshold=0.85
        )
        
        # Process factors with advanced techniques
        all_factors, processing_results = enhanced_processor.process_factors_enhanced(
            raw_factors, price_data, top_n=40
        )
        
        if all_factors.empty:
            print("Warning: Enhanced processing returned no factors, using raw factors")
            all_factors = raw_factors
            processing_results = {}
        else:
            print(f"Enhanced processing successful: {len(all_factors.columns)} high-quality factors")
            
        logger.info(f"Final factors after enhancement: {len(all_factors.columns)}")
        
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
        
        # Step 5.5: Apply advanced quality control
        print("\n5. Applying advanced quality control...")
        quality_controller = FactorQualityController(
            min_ic_threshold=0.005,
            min_observations=50,
            max_missing_ratio=0.2,
            correlation_threshold=0.85,
            outlier_contamination=0.1
        )
        
        # Filter factors by quality
        high_quality_factors, quality_results = quality_controller.filter_factors_by_quality(
            all_factors, ic_results, min_quality_score=0.1
        )
        
        # Generate quality report
        quality_report = quality_controller.get_quality_report()
        
        print(f"\nQuality Control Results:")
        print(f"  Total factors: {quality_report['summary']['total_factors']}")
        print(f"  Valid factors: {quality_report['summary']['valid_factors']}")
        print(f"  High quality: {quality_report['summary']['high_quality']}")
        print(f"  Medium quality: {quality_report['summary']['medium_quality']}")
        print(f"  Low quality: {quality_report['summary']['low_quality']}")
        
        # Debug: Show some quality scores
        print(f"\nSample Quality Scores:")
        quality_scores = quality_controller.quality_scores
        sorted_scores = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for factor, score in sorted_scores:
            print(f"  {factor}: {score:.3f}")
        
        # Use high-quality factors for portfolio simulation
        if len(high_quality_factors.columns) > 0:
            print(f"\nUsing {len(high_quality_factors.columns)} high-quality factors for portfolio simulation")
            all_factors = high_quality_factors
        else:
            print("\nWarning: No high-quality factors found. Using original factors.")
        
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
                    print(f"    {count} factors")
        else:
            print("\nNo factors met the enhanced screening criteria.")
            print("Consider relaxing thresholds or improving factor quality.")
        
        # Step 6: Cross-validation and out-of-sample testing
        print("\n6. Running cross-validation and out-of-sample testing...")
        
        # Create returns DataFrame for validation
        returns_df = pd.DataFrame()
        for ticker in ['SPY', 'MTUM', 'VLUE', 'QUAL', 'USMV', 'SIZE']:
            if ticker in price_data:
                if isinstance(price_data[ticker].columns, pd.MultiIndex):
                    price_col = ('Adj Close', ticker) if ('Adj Close', ticker) in price_data[ticker].columns else ('Close', ticker)
                else:
                    price_col = 'Adj Close' if 'Adj Close' in price_data[ticker].columns else 'Close'
                
                returns_df[ticker] = price_data[ticker][price_col].pct_change().shift(-1)  # Forward returns
        
        # Remove rows with all NaN returns
        returns_df = returns_df.dropna(how='all')
        
        # Align factors and returns
        common_dates = all_factors.index.intersection(returns_df.index)
        factors_aligned = all_factors.loc[common_dates]
        returns_aligned = returns_df.loc[common_dates]
        
        print(f"  Aligned data: {len(factors_aligned)} observations")
        
        # Time series cross-validation
        print("\n6.1. Time series cross-validation...")
        cv_validator = TimeSeriesCrossValidator(n_splits=3, test_size=0.2)
        cv_results = cv_validator.validate_factor_model(
            factors_aligned, returns_aligned, 
            validate_factor_model_comprehensive
        )
        
        # Walk-forward analysis
        print("\n6.2. Walk-forward analysis...")
        wf_validator = WalkForwardValidator(train_window=252, test_window=63, step_size=63)
        wf_results = wf_validator.walk_forward_analysis(
            factors_aligned, returns_aligned,
            validate_factor_model_comprehensive
        )
        
        # Out-of-sample testing
        print("\n6.3. Out-of-sample testing...")
        oos_tester = OutOfSampleTester(train_ratio=0.6, validation_ratio=0.2)
        oos_results = oos_tester.test_model_generalization(
            factors_aligned, returns_aligned,
            validate_factor_model_single_dataset
        )
        
        # Factor stability analysis
        print("\n6.4. Factor stability analysis...")
        stability_results = validate_factor_stability(factors_aligned, returns_aligned)
        
        # Step 7: Simulate portfolio
        print("\n7. Simulating portfolio...")
        # Use the actual factor values, not the IC statistics
        portfolio_sim = PortfolioSimulator(all_factors, price_data)
        portfolio_results = portfolio_sim.simulate()
        logger.info("Portfolio simulation completed")
        
        # Step 8: Save results
        print("\n8. Saving results...")
        results = {
            'ic_results': ic_results,
            'top_factors': top_factors,
            'portfolio_results': portfolio_results,
            'all_factors': all_factors,
            'quality_results': quality_results,
            'quality_report': quality_report,
            'processing_results': processing_results,
            'cross_validation_results': cv_results,
            'walk_forward_results': wf_results,
            'out_of_sample_results': oos_results,
            'stability_results': stability_results
        }
        save_results(results)
        logger.info("Results saved successfully")
        
        # Step 9: Print summary
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Total factors analyzed: {len(all_factors.columns)}")
        print(f"Top factors selected: {len(top_factors)}")
        print(f"Portfolio Sharpe ratio: {portfolio_results.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"Portfolio total return: {portfolio_results.get('total_return', 'N/A'):.1%}")
        
        # Print validation summary
        print(f"\nVALIDATION RESULTS:")
        print(f"  Cross-validation folds: {len(cv_results.get('fold_results', []))}")
        print(f"  Walk-forward windows: {wf_results.get('window_count', 0)}")
        print(f"  Out-of-sample test: {oos_results.get('generalization_analysis', {}).get('overall_assessment', {}).get('assessment', 'Unknown')}")
        print(f"  Factor stability: {stability_results.get('factor_stability', {}).get('stability_assessment', 'Unknown')}")
        
        # Print overfitting assessment
        if 'overfitting_analysis' in cv_results:
            ic_degradation = cv_results['overfitting_analysis'].get('ic_overfitting', {}).get('ic_degradation', 0)
            sharpe_degradation = cv_results['overfitting_analysis'].get('sharpe_overfitting', {}).get('sharpe_degradation', 0)
            
            print(f"\nOVERFITTING ASSESSMENT:")
            if abs(ic_degradation) < 0.01 and abs(sharpe_degradation) < 0.1:
                print(f"  ✅ LOW OVERFITTING: Model generalizes well")
            elif abs(ic_degradation) < 0.03 and abs(sharpe_degradation) < 0.3:
                print(f"  ⚠️  MODERATE OVERFITTING: Some degradation observed")
            else:
                print(f"  ❌ HIGH OVERFITTING: Significant degradation observed")
            
            print(f"  IC Degradation: {ic_degradation:.4f}")
            print(f"  Sharpe Degradation: {sharpe_degradation:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\nERROR: Pipeline failed - {str(e)}")
        raise
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
