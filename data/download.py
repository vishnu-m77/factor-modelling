#!/usr/bin/env python3
"""
Data Download and Preprocessing Module
Downloads free financial data and preprocesses for factor analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class DataDownloader:
    """Downloads and preprocesses financial data from free sources"""
    
    def __init__(self, start_date='2015-01-01', end_date=None):
        """
        Initialize data downloader
        
        Args:
            start_date (str): Start date for data download
            end_date (str): End date for data download (defaults to today)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Default tickers for factor analysis
        self.tickers = ['SPY', 'MTUM', 'VLUE', 'QUAL', 'USMV', 'SIZE']
        self.cache_dir = 'cache'
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def download_price_data(self):
        """Download price data for all tickers"""
        print(f"Downloading price data from {self.start_date} to {self.end_date}")
        
        price_data = {}
        for ticker in self.tickers:
            try:
                print(f"  Downloading {ticker}...")
                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if not ticker_data.empty:
                    # Calculate returns
                    ticker_data['Returns'] = ticker_data['Adj Close'].pct_change()
                    ticker_data['Log_Returns'] = np.log(ticker_data['Adj Close'] / ticker_data['Adj Close'].shift(1))
                    
                    # Calculate volatility (rolling 20-day)
                    ticker_data['Volatility'] = ticker_data['Returns'].rolling(20).std()
                    
                    price_data[ticker] = ticker_data
                    print(f"    {ticker}: {len(ticker_data)} observations")
                else:
                    print(f"    {ticker}: No data available")
                    
            except Exception as e:
                print(f"    {ticker}: Error downloading - {str(e)}")
                continue
        
        return price_data
    
    def create_fundamental_data(self, price_data):
        """Create synthetic fundamental data based on price data"""
        print("Creating fundamental data proxies...")
        
        fundamental_data = {}
        
        for ticker, data in price_data.items():
            if data.empty:
                continue
                
            # Create fundamental proxies based on price data
            fundamental = pd.DataFrame(index=data.index)
            
            # Price-based ratios
            fundamental['Price_Ratio'] = data['Adj Close'] / data['Adj Close'].rolling(252).mean()
            fundamental['Price_Momentum'] = data['Adj Close'] / data['Adj Close'].shift(20) - 1
            
            # Volume-based metrics
            if 'Volume' in data.columns:
                fundamental['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
                fundamental['Volume_Momentum'] = data['Volume'] / data['Volume'].shift(5) - 1
            
            # Volatility-based metrics
            fundamental['Volatility_Ratio'] = data['Volatility'] / data['Volatility'].rolling(252).mean()
            fundamental['Volatility_Momentum'] = data['Volatility'] / data['Volatility'].shift(20) - 1
            
            # Size proxy (using price as proxy for market cap)
            fundamental['Size_Proxy'] = np.log(data['Adj Close'])
            
            # Quality proxy (using volatility as inverse quality measure)
            fundamental['Quality_Proxy'] = 1 / (1 + data['Volatility'])
            
            fundamental_data[ticker] = fundamental
        
        return fundamental_data
    
    def align_data(self, price_data, fundamental_data):
        """Align price and fundamental data on common dates"""
        print("Aligning data on common dates...")
        
        # Get common dates
        common_dates = None
        for ticker in price_data.keys():
            if common_dates is None:
                common_dates = set(price_data[ticker].index)
            else:
                common_dates = common_dates.intersection(set(price_data[ticker].index))
        
        common_dates = sorted(list(common_dates))
        print(f"  Common dates: {len(common_dates)} observations")
        
        # Align price data
        aligned_price = {}
        for ticker, data in price_data.items():
            aligned_price[ticker] = data.loc[common_dates]
        
        # Align fundamental data
        aligned_fundamental = {}
        for ticker, data in fundamental_data.items():
            if ticker in aligned_price:
                aligned_fundamental[ticker] = data.loc[common_dates]
        
        return aligned_price, aligned_fundamental
    
    def validate_data(self, price_data, fundamental_data):
        """Validate data quality and handle missing values"""
        print("Validating data quality...")
        
        # Check for missing values
        total_missing = 0
        for ticker, data in price_data.items():
            missing = data.isnull().sum().sum()
            total_missing += missing
            if missing > 0:
                print(f"  {ticker}: {missing} missing values")
        
        # Forward fill missing values for price data
        for ticker in price_data.keys():
            price_data[ticker] = price_data[ticker].fillna(method='ffill')
            price_data[ticker] = price_data[ticker].fillna(method='bfill')
        
        # Forward fill missing values for fundamental data
        for ticker in fundamental_data.keys():
            fundamental_data[ticker] = fundamental_data[ticker].fillna(method='ffill')
            fundamental_data[ticker] = fundamental_data[ticker].fillna(method='bfill')
        
        print(f"  Total missing values handled: {total_missing}")
        
        return price_data, fundamental_data
    
    def save_to_cache(self, price_data, fundamental_data):
        """Save data to cache for future use"""
        print("Saving data to cache...")
        
        # Save price data
        for ticker, data in price_data.items():
            cache_file = os.path.join(self.cache_dir, f"{ticker}_prices.csv")
            data.to_csv(cache_file)
            print(f"  Saved {ticker} prices to {cache_file}")
        
        # Save fundamental data
        for ticker, data in fundamental_data.items():
            cache_file = os.path.join(self.cache_dir, f"{ticker}_fundamentals.csv")
            data.to_csv(cache_file)
            print(f"  Saved {ticker} fundamentals to {cache_file}")
    
    def run(self):
        """Main method to run the data download pipeline"""
        print("Starting data download pipeline...")
        
        # Download price data
        price_data = self.download_price_data()
        
        if not price_data:
            raise ValueError("No price data downloaded")
        
        # Create fundamental data
        fundamental_data = self.create_fundamental_data(price_data)
        
        # Align data
        aligned_price, aligned_fundamental = self.align_data(price_data, fundamental_data)
        
        # Validate data
        validated_price, validated_fundamental = self.validate_data(aligned_price, aligned_fundamental)
        
        # Save to cache
        self.save_to_cache(validated_price, validated_fundamental)
        
        print("Data download pipeline completed successfully")
        
        return validated_price, validated_fundamental

def main():
    """Test the data downloader"""
    downloader = DataDownloader()
    price_data, fundamental_data = downloader.run()
    
    print(f"\nDownloaded data summary:")
    print(f"Price data: {len(price_data)} tickers")
    print(f"Fundamental data: {len(fundamental_data)} tickers")
    
    if price_data:
        sample_ticker = list(price_data.keys())[0]
        print(f"\nSample data for {sample_ticker}:")
        print(price_data[sample_ticker].head())

if __name__ == "__main__":
    main() 