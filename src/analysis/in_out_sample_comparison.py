#!/usr/bin/env python3
"""
In-Sample vs Out-of-Sample Performance Comparison
Focused on 3 portfolios: High Relevance, Low Relevance, S&P 500

Author: Battle of Quants Team  
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InOutSampleAnalyzer:
    """
    Compare in-sample vs out-of-sample performance for 3 portfolios:
    - High Relevance (>=0.75)
    - Low Relevance (<0.3)
    - S&P 500
    """
    
    def __init__(self, 
                 trades_df: pd.DataFrame,
                 split_date: str = '2020-01-01',
                 holding_period_days: int = 126,
                 price_data_file: str = None):
        """
        Initialize analyzer with train/test split
        
        Args:
            trades_df: Full dataset of trades with relevance scores
            split_date: Date to split in-sample/out-of-sample
            holding_period_days: 6-month holding period (126 trading days)
            price_data_file: Path to price data parquet
        """
        self.trades_df = trades_df.copy()
        self.split_date = pd.to_datetime(split_date)
        self.holding_period_days = holding_period_days
        self.price_data_file = price_data_file
        
        # Split data
        self.in_sample = trades_df[trades_df['Traded'] < self.split_date].copy()
        self.out_sample = trades_df[trades_df['Traded'] >= self.split_date].copy()
        
        logger.info(f"Split at {split_date}:")
        logger.info(f"  In-sample: {len(self.in_sample):,} trades ({self.in_sample['Traded'].min().date()} to {self.in_sample['Traded'].max().date()})")
        logger.info(f"  Out-sample: {len(self.out_sample):,} trades ({self.out_sample['Traded'].min().date()} to {self.out_sample['Traded'].max().date()})")
        
        # Load S&P 500 benchmark data
        self.sp500_in_sample = None
        self.sp500_out_sample = None
        self._load_sp500_benchmark()
    
    def _load_sp500_benchmark(self):
        """
        Load S&P 500 benchmark data for in-sample and out-of-sample periods
        Uses same logic as visualizations.py
        """
        logger.info("\nLoading S&P 500 benchmark data...")
        
        # Try multiple sources
        # 1. Try Russell 3000 data (if available)
        if self.price_data_file and Path(self.price_data_file).exists():
            try:
                logger.info("  Attempting to use Russell 3000 as market proxy...")
                price_data = pd.read_parquet(self.price_data_file)
                
                if 'return' in price_data.columns:
                    # Calculate equal-weighted market returns
                    price_data['date'] = pd.to_datetime(price_data['date'])
                    market_returns = price_data.groupby('date')['return'].mean().reset_index()
                    market_returns.columns = ['date', 'return']
                    market_returns['return'] = market_returns['return'] / 100.0  # Convert to decimal
                    
                    # Resample to monthly
                    market_returns = market_returns.set_index('date')
                    market_monthly = market_returns['return'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
                    market_monthly = market_monthly.reset_index()
                    market_monthly.columns = ['date', 'return']
                    
                    # Split into in-sample and out-of-sample
                    self.sp500_in_sample = market_monthly[market_monthly['date'] < self.split_date].copy()
                    self.sp500_out_sample = market_monthly[market_monthly['date'] >= self.split_date].copy()
                    
                    logger.info(f"  ✓ Loaded Russell 3000 market proxy:")
                    logger.info(f"    In-sample: {len(self.sp500_in_sample)} months")
                    logger.info(f"    Out-sample: {len(self.sp500_out_sample)} months")
                    return
                    
            except Exception as e:
                logger.warning(f"  Could not load Russell 3000 data: {e}")
        
        # 2. Download SPY data
        try:
            logger.info("  Downloading SPY data from yfinance...")
            
            start_date = self.in_sample['Traded'].min() - timedelta(days=30)
            end_date = self.out_sample['Traded'].max() + timedelta(days=30)
            
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            if len(spy) > 0:
                spy['Returns'] = spy['Adj Close'].pct_change()
                spy_monthly = spy['Returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
                spy_monthly = spy_monthly.reset_index()
                spy_monthly.columns = ['date', 'return']
                
                # Split
                self.sp500_in_sample = spy_monthly[spy_monthly['date'] < self.split_date].copy()
                self.sp500_out_sample = spy_monthly[spy_monthly['date'] >= self.split_date].copy()
                
                logger.info(f"  ✓ Downloaded SPY benchmark:")
                logger.info(f"    In-sample: {len(self.sp500_in_sample)} months")
                logger.info(f"    Out-sample: {len(self.sp500_out_sample)} months")
                return
                
        except Exception as e:
            logger.warning(f"  Could not download SPY: {e}")
        
        # 3. Fallback: Use historical average
        logger.warning("  Using historical S&P 500 average (12% annual) as fallback")
        
        # Create synthetic benchmark
        in_months = int((self.split_date - self.in_sample['Traded'].min()).days / 30)
        out_months = int((self.out_sample['Traded'].max() - self.split_date).days / 30)
        
        monthly_return = 0.12 / 12  # 12% annual = 1% monthly
        
        in_dates = pd.date_range(start=self.in_sample['Traded'].min(), periods=in_months, freq='ME')
        out_dates = pd.date_range(start=self.split_date, periods=out_months, freq='ME')
        
        self.sp500_in_sample = pd.DataFrame({
            'date': in_dates,
            'return': [monthly_return] * len(in_dates)
        })
        
        self.sp500_out_sample = pd.DataFrame({
            'date': out_dates,
            'return': [monthly_return] * len(out_dates)
        })
        
        logger.info(f"  Using synthetic benchmark: 12% annual")
    
    def run_comparative_analysis(self, portfolios_config: Dict) -> Dict:
        """
        Run comprehensive in-sample vs out-of-sample comparison
        
        Args:
            portfolios_config: Dictionary defining portfolio configurations
        
        Returns:
            Dictionary with comparative results
        """
        logger.info("\n" + "="*80)
        logger.info("IN-SAMPLE VS OUT-OF-SAMPLE ANALYSIS")
        logger.info("="*80)
        
        from calendar_time_portfolio_v4 import CalendarTimePortfolioV2
        
        results = {
            'in_sample': {},
            'out_sample': {},
            'comparison': {}
        }
        
        # Run in-sample analysis
        logger.info("\n1. IN-SAMPLE ANALYSIS")
        logger.info("-" * 80)
        
        analyzer_in = CalendarTimePortfolioV2(
            self.in_sample, 
            holding_period_days=self.holding_period_days,
            price_data_file=self.price_data_file
        )
        analyzer_in.load_factor_data()
        
        for name, config in portfolios_config.items():
            logger.info(f"\nAnalyzing: {name} (In-Sample)")
            
            # Filter portfolio based on config
            portfolio_trades = self._filter_portfolio(self.in_sample, config)
            
            if len(portfolio_trades) < 50:
                logger.warning(f"Skipping {name}: only {len(portfolio_trades)} trades")
                continue
            
            result = analyzer_in.analyze_portfolio(name, portfolio_trades)
            results['in_sample'][name] = result
        
        # Add S&P 500 in-sample
        if self.sp500_in_sample is not None:
            results['in_sample']['S&P 500'] = self._create_sp500_result(
                self.sp500_in_sample, 
                'S&P 500', 
                'in_sample'
            )
        
        # Run out-of-sample analysis
        logger.info("\n2. OUT-OF-SAMPLE ANALYSIS")
        logger.info("-" * 80)
        
        analyzer_out = CalendarTimePortfolioV2(
            self.out_sample,
            holding_period_days=self.holding_period_days,
            price_data_file=self.price_data_file
        )
        analyzer_out.load_factor_data()
        
        for name, config in portfolios_config.items():
            logger.info(f"\nAnalyzing: {name} (Out-of-Sample)")
            
            portfolio_trades = self._filter_portfolio(self.out_sample, config)
            
            if len(portfolio_trades) < 50:
                logger.warning(f"Skipping {name}: only {len(portfolio_trades)} trades")
                continue
            
            result = analyzer_out.analyze_portfolio(name, portfolio_trades)
            results['out_sample'][name] = result
        
        # Add S&P 500 out-of-sample
        if self.sp500_out_sample is not None:
            results['out_sample']['S&P 500'] = self._create_sp500_result(
                self.sp500_out_sample,
                'S&P 500',
                'out_sample'
            )
        
        # Comparative analysis
        logger.info("\n3. COMPARATIVE ANALYSIS")
        logger.info("-" * 80)
        
        results['comparison'] = self._compare_samples(
            results['in_sample'],
            results['out_sample']
        )
        
        return results
    
    def _create_sp500_result(self, sp500_data: pd.DataFrame, name: str, sample_type: str) -> Dict:
        """
        Create result dictionary for S&P 500 benchmark
        """
        returns = sp500_data['return'].values
        
        # Calculate metrics
        total_return = (np.prod(1 + returns) - 1)
        n_months = len(returns)
        annualized_return = ((1 + total_return) ** (12/n_months) - 1) if n_months > 0 else 0
        
        volatility = np.std(returns) * np.sqrt(12)
        risk_free = 0.02  # 2% annual
        sharpe = (annualized_return - risk_free) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        
        return {
            'portfolio_name': name,
            'n_trades': 0,
            'monthly_returns': sp500_data,
            'capm': {
                'alpha_annual': 0.0,  # S&P 500 has zero alpha by definition
                't_stat': 0.0,
                'p_value': 1.0,
                'n_months': n_months
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe,
                'information_ratio': 0.0,
                'max_drawdown': max_dd,
                'volatility_annual': volatility,
                'total_return': total_return
            }
        }
    
    def _filter_portfolio(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Filter trades based on portfolio configuration"""
        filtered = df.copy()
        
        if 'relevance_threshold' in config:
            threshold = config['relevance_threshold']
            comparison = config.get('comparison', '>=')
            
            if comparison == '>=':
                filtered = filtered[filtered['relevance_score'] >= threshold]
            elif comparison == '<':
                filtered = filtered[filtered['relevance_score'] < threshold]
        
        if 'chamber' in config:
            filtered = filtered[filtered['Chamber'] == config['chamber']]
        
        if 'direction' in config:
            filtered = filtered[filtered['direction'] == config['direction']]
        
        return filtered
    
    def _compare_samples(self, in_results: Dict, out_results: Dict) -> Dict:
        """
        Compare in-sample vs out-of-sample performance
        """
        comparison = {}
        
        for portfolio_name in in_results.keys():
            if portfolio_name not in out_results:
                continue
            
            in_res = in_results[portfolio_name]
            out_res = out_results[portfolio_name]
            
            # Extract key metrics
            in_capm = in_res.get('capm', {})
            out_capm = out_res.get('capm', {})
            
            in_risk = in_res.get('risk_metrics', {})
            out_risk = out_res.get('risk_metrics', {})
            
            comparison[portfolio_name] = {
                'in_sample': {
                    'alpha_annual': in_capm.get('alpha_annual', 0),
                    't_stat': in_capm.get('t_stat', 0),
                    'p_value': in_capm.get('p_value', 1),
                    'sharpe': in_risk.get('sharpe_ratio', 0),
                    'max_dd': in_risk.get('max_drawdown', 0),
                    'n_trades': in_res.get('n_trades', 0),
                    'n_months': in_capm.get('n_months', 0),
                    'monthly_returns': in_res.get('monthly_returns')
                },
                'out_sample': {
                    'alpha_annual': out_capm.get('alpha_annual', 0),
                    't_stat': out_capm.get('t_stat', 0),
                    'p_value': out_capm.get('p_value', 1),
                    'sharpe': out_risk.get('sharpe_ratio', 0),
                    'max_dd': out_risk.get('max_drawdown', 0),
                    'n_trades': out_res.get('n_trades', 0),
                    'n_months': out_capm.get('n_months', 0),
                    'monthly_returns': out_res.get('monthly_returns')
                }
            }
        
        return comparison


def main():
    """Run in-sample vs out-of-sample analysis"""
    
    print("="*80)
    print("IN-SAMPLE VS OUT-OF-SAMPLE ANALYSIS")
    print("3 Portfolios: High Relevance, Low Relevance, S&P 500")
    print("="*80)
    
    # Load data
    df = pd.read_pickle('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_relevance_v2_quality.pkl')
    
    # Filter to price data availability period
    price_data_start = pd.to_datetime('2013-07-31')
    df = df[df['Traded'] >= price_data_start]
    
    print(f"\nLoaded {len(df):,} trades")
    print(f"Date range: {df['Traded'].min().date()} to {df['Traded'].max().date()}")
    
    # Initialize analyzer with 2020 split
    analyzer = InOutSampleAnalyzer(
        trades_df=df,
        split_date='2020-01-01',
        holding_period_days=126,
        price_data_file='/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw/russell_3000_daily.parquet'
    )
    
    # Define ONLY the 3 portfolios we want
    portfolios = {
        'High Relevance (≥0.75)': {
            'relevance_threshold': 0.75,
            'comparison': '>=',
            'direction': 1
        },
        'Low Relevance (<0.3)': {
            'relevance_threshold': 0.3,
            'comparison': '<',
            'direction': 1
        }
        # S&P 500 added automatically in run_comparative_analysis
    }
    
    # Run analysis
    results = analyzer.run_comparative_analysis(portfolios)
    
    # Save results
    import pickle
    output_dir = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed'
    
    with open(f'{output_dir}/in_out_sample_results_v2.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {output_dir}/in_out_sample_results_v2.pkl")
    
    return results


if __name__ == "__main__":
    results = main()