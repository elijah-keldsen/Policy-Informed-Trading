#!/usr/bin/env python3
"""
Calendar-Time Portfolio Analysis, vectorized operation-->improved performance

Author: ShaneStreet
Date: 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import time
import pickle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalendarTimePortfolioV2:
    """
    Calendar-time portfolio with optimized vectorized operations
    """
    
    def __init__(self, trades_df: pd.DataFrame, holding_period_days: int = 126, 
                 price_data_file: Optional[str] = None):
        self.trades_df = trades_df.copy()
        self.holding_period_days = holding_period_days
        
        self.trades_df['Traded'] = pd.to_datetime(self.trades_df['Traded'])
        
        # Load price data from parquet
        self.price_data_df = None
        self.returns_pivot = None
        
        if price_data_file and os.path.exists(price_data_file):
            logger.info(f"Loading price data from: {price_data_file}")
            self.price_data_df = pd.read_parquet(price_data_file)
            logger.info(f"Loaded {len(self.price_data_df):,} rows of price data")
            
            if 'date' in self.price_data_df.columns:
                self.price_data_df['date'] = pd.to_datetime(self.price_data_df['date'])
                logger.info(f"Date range: {self.price_data_df['date'].min()} to {self.price_data_df['date'].max()}")
                logger.info(f"Unique tickers: {self.price_data_df['ticker'].nunique():,}")
            
            if 'return' in self.price_data_df.columns:
                logger.info("Pre-calculated returns available")
            elif 'price' in self.price_data_df.columns:
                logger.info("Price data available, will calculate returns")
        
        self.factor_data = None
        
        logger.info(f"Calendar-Time Portfolio initialized with {len(self.trades_df):,} trades")
        logger.info(f"Date range: {self.trades_df['Traded'].min().date()} to {self.trades_df['Traded'].max().date()}")
    
    def load_factor_data(self):
        """
        Load factor data
        
        Academic justification (Ziobrowski et al. 2004):
        - Ziobrowski used CRSP value-weighted index as market benchmark
        - Russell 3000 represents approximately 98% of US equity market
        """
        logger.info("Loading factor data...")
        
        start_date = self.trades_df['Traded'].min() - timedelta(days=365)
        end_date = datetime.now()
        
        # Option 1: Create equal-weighted market index from Russell 3000 data
        if self.price_data_df is not None:
            logger.info("Creating equal-weighted market index from Russell 3000...")
            
            try:
                # Filter to date range
                market_data = self.price_data_df[
                    (self.price_data_df['date'] >= start_date) &
                    (self.price_data_df['date'] <= end_date)
                ].copy()
                
                logger.info(f"Processing {len(market_data):,} rows across {market_data['ticker'].nunique():,} stocks")
                
                # Convert returns from percentage to decimal
                if 'return' in market_data.columns:
                    market_data['return_decimal'] = market_data['return'] / 100.0
                elif 'price' in market_data.columns:
                    market_data = market_data.sort_values(['ticker', 'date'])
                    market_data['return_decimal'] = market_data.groupby('ticker')['price'].pct_change()
                
                # Calculate daily equal-weighted market return
                daily_market = market_data.groupby('date')['return_decimal'].mean()
                
                # Compound to monthly returns (Ziobrowski method, page 665)
                daily_market_df = pd.DataFrame({'return': daily_market})
                daily_market_df['year_month'] = daily_market_df.index.to_period('M')
                
                monthly_returns = []
                for period, group in daily_market_df.groupby('year_month'):
                    # Geometric compounding
                    monthly_return = (1 + group['return']).prod() - 1
                    monthly_returns.append({
                        'date': period.to_timestamp('M'),
                        'market_return': monthly_return
                    })
                
                monthly_df = pd.DataFrame(monthly_returns)
                
                # Create risk-free rate (2% annual = 0.167% monthly)
                rf_monthly = 0.02 / 12
                
                factor_df = pd.DataFrame({
                    'date': monthly_df['date'],
                    'mkt_rf': monthly_df['market_return'] - rf_monthly,
                    'rf': rf_monthly,
                    'smb': np.nan,
                    'hml': np.nan,
                    'mom': np.nan
                })
                
                self.factor_data = factor_df
                
                logger.info(f"Created market factor: {len(factor_df)} months")
                logger.info(f"Note: FF3 factors (SMB/HML) not available - CAPM only")
                return
                
            except Exception as e:
                logger.error(f"Could not create market index from parquet: {e}")
                import traceback
                traceback.print_exc()
        
        # Option 2: Try yfinance SPY
        logger.info("Attempting to download SPY from yfinance...")
        try:
            time.sleep(2)
            
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            if len(spy) == 0:
                raise ValueError("No SPY data returned")
            
            spy['Returns'] = spy['Adj Close'].pct_change()
            spy_monthly = spy['Returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
            
            rf_monthly = pd.Series(0.02/12, index=spy_monthly.index)
            
            factor_df = pd.DataFrame({
                'date': spy_monthly.index,
                'mkt_rf': spy_monthly.values - rf_monthly.values,
                'rf': rf_monthly.values,
                'smb': np.nan,
                'hml': np.nan,
                'mom': np.nan
            })
            
            self.factor_data = factor_df
            
            logger.info(f"Downloaded SPY: {len(factor_df)} months")
            
        except Exception as e:
            logger.error(f"Could not load any market data: {e}")
    
    def create_returns_matrix(self, tickers: List[str], start_date: datetime, end_date: datetime):
        """
        Create a date by ticker matrix of returns for vectorized lookups
        """
        if self.price_data_df is None:
            return None
        
        logger.info(f"Creating returns matrix for {len(tickers)} tickers...")
        
        # Filter data
        filtered_data = self.price_data_df[
            (self.price_data_df['ticker'].isin(tickers)) &
            (self.price_data_df['date'] >= start_date) &
            (self.price_data_df['date'] <= end_date)
        ].copy()
        
        # Convert returns from percentage to decimal
        if 'return' in filtered_data.columns:
            filtered_data['return_decimal'] = filtered_data['return'] / 100.0
        elif 'price' in filtered_data.columns:
            filtered_data = filtered_data.sort_values(['ticker', 'date'])
            filtered_data['return_decimal'] = filtered_data.groupby('ticker')['price'].pct_change()
        else:
            return None
        
        # Pivot to create date by ticker matrix
        returns_matrix = filtered_data.pivot(index='date', columns='ticker', values='return_decimal')
        
        logger.info(f"Created matrix: {returns_matrix.shape[0]:,} dates x {returns_matrix.shape[1]:,} tickers")
        
        return returns_matrix
    
    def build_calendar_time_portfolio_vectorized(self, 
                                                 trades_subset: pd.DataFrame,
                                                 weight_method: str = 'equal') -> pd.DataFrame:
        """
        Build calendar-time portfolio using vectorized operations
        """
        logger.info(f"Building calendar-time portfolio for {len(trades_subset):,} trades...")
        
        start_date = trades_subset['Traded'].min() - timedelta(days=30)
        end_date = min(datetime.now(), trades_subset['Traded'].max() + timedelta(days=self.holding_period_days + 30))
        
        # Create returns matrix once
        unique_tickers = trades_subset['Ticker'].unique()
        returns_matrix = self.create_returns_matrix(unique_tickers, start_date, end_date)
        
        if returns_matrix is None or returns_matrix.empty:
            logger.error("Could not create returns matrix")
            return pd.DataFrame()
        
        # Create date range
        calendar_days = pd.date_range(start=trades_subset['Traded'].min(), end=end_date, freq='D')
        
        logger.info(f"Building portfolio across {len(calendar_days):,} days...")
        
        portfolio_returns = []
        
        for idx, current_date in enumerate(calendar_days):
            if idx % 500 == 0 and idx > 0:
                pct = (idx / len(calendar_days)) * 100
                logger.info(f"Progress: {pct:.0f}% ({idx:,}/{len(calendar_days):,} days)")
            
            holding_window_start = current_date - timedelta(days=self.holding_period_days)
            
            # Get active trades
            active_trades = trades_subset[
                (trades_subset['Traded'] >= holding_window_start) &
                (trades_subset['Traded'] <= current_date)
            ]
            
            if len(active_trades) == 0:
                portfolio_returns.append({
                    'date': current_date,
                    'return': 0.0,
                    'n_holdings': 0
                })
                continue
            
            # Get returns for this date - vectorized
            if current_date not in returns_matrix.index:
                portfolio_returns.append({
                    'date': current_date,
                    'return': 0.0,
                    'n_holdings': 0
                })
                continue
            
            day_returns = returns_matrix.loc[current_date]
            
            # Calculate portfolio return vectorized
            valid_positions = []
            for _, trade in active_trades.iterrows():
                ticker = trade['Ticker']
                direction = trade['direction']
                
                if ticker in day_returns.index and not pd.isna(day_returns[ticker]):
                    if weight_method == 'equal':
                        weight = 1.0
                    else:
                        weight = trade.get('amount_midpoint', 25000)
                    
                    valid_positions.append({
                        'return': day_returns[ticker] * direction,
                        'weight': weight
                    })
            
            if len(valid_positions) > 0:
                weights = np.array([p['weight'] for p in valid_positions])
                returns = np.array([p['return'] for p in valid_positions])
                weights = weights / weights.sum()
                portfolio_return = np.sum(returns * weights)
                n_holdings = len(valid_positions)
            else:
                portfolio_return = 0.0
                n_holdings = 0
            
            portfolio_returns.append({
                'date': current_date,
                'return': portfolio_return,
                'n_holdings': n_holdings
            })
        
        portfolio_df = pd.DataFrame(portfolio_returns)
        days_with_data = (portfolio_df['n_holdings'] > 0).sum()
        
        logger.info(f"Portfolio complete: {len(portfolio_df):,} days, {days_with_data:,} days with holdings")
        
        return portfolio_df
    
    def calculate_monthly_returns(self, daily_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily returns to monthly"""
        daily_portfolio = daily_portfolio.copy()
        
        daily_portfolio['year_month'] = pd.to_datetime(daily_portfolio['date']).dt.to_period('M')
        
        monthly_returns = []
        
        for period, group in daily_portfolio.groupby('year_month'):
            monthly_return = (1 + group['return']).prod() - 1
            
            # Use period end date to match factor data format
            monthly_returns.append({
                'date': period.to_timestamp('M'),
                'return': monthly_return,
                'n_days': len(group),
                'avg_holdings': group['n_holdings'].mean()
            })
        
        monthly_df = pd.DataFrame(monthly_returns)
        
        return monthly_df
    
    def run_factor_regression(self, monthly_returns: pd.DataFrame, model: str = 'ff3') -> Dict:
        """Run factor regression with proper NaN handling for each model"""
        from scipy import stats
        
        if self.factor_data is None:
            logger.error("Factor data is None")
            return {'alpha': 0.0, 'alpha_annual': 0.0, 't_stat': 0.0, 'p_value': 1.0, 'n_months': 0}
        
        # Merge monthly returns with factor data
        merged = monthly_returns.merge(self.factor_data, on='date', how='inner')
        
        if len(merged) == 0:
            logger.error("Merge failed - 0 rows after merge")
            return {'alpha': 0.0, 'alpha_annual': 0.0, 't_stat': 0.0, 'p_value': 1.0, 'n_months': 0}
        
        if len(merged) < 12:
            logger.warning(f"Only {len(merged)} months - results may be unreliable")
        
        # Calculate excess return
        merged['excess_return'] = merged['return'] - merged['rf']
        
        # Clean based on the factors we're actually using
        if model == 'capm':
            merged = merged.replace([np.inf, -np.inf], np.nan)
            cols_to_check = ['excess_return', 'mkt_rf']
            merged = merged.dropna(subset=cols_to_check)
            
            if len(merged) == 0:
                logger.error("All rows dropped after CAPM cleaning")
                return {'alpha': 0.0, 'alpha_annual': 0.0, 't_stat': 0.0, 'p_value': 1.0, 'n_months': 0}
            
            y = merged['excess_return'].values
            X = merged[['mkt_rf']].values
            factor_names = ['mkt_rf']
            
        else:  # ff3
            # Check if SMB/HML are available
            if merged['smb'].isna().all() or merged['hml'].isna().all():
                logger.warning("SMB/HML factors not available, skipping FF3")
                return {
                    'alpha': 0, 
                    'alpha_annual': 0, 
                    't_stat': 0, 
                    'p_value': 0, 
                    'n_months': 0, 
                    'note': 'FF3 skipped'
                }
            
            merged = merged.replace([np.inf, -np.inf], np.nan)
            cols_to_check = ['excess_return', 'mkt_rf', 'smb', 'hml']
            merged = merged.dropna(subset=cols_to_check)
            
            if len(merged) == 0:
                logger.error("All rows dropped after FF3 cleaning")
                return {'alpha': 0.0, 'alpha_annual': 0.0, 't_stat': 0.0, 'p_value': 1.0, 'n_months': 0}
            
            y = merged['excess_return'].values
            X = merged[['mkt_rf', 'smb', 'hml']].values
            factor_names = ['mkt_rf', 'smb', 'hml']
        
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        
        try:
            # Run regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            
            # Calculate statistics
            n, k = len(y), X.shape[1]
            residual_variance = np.sum(residuals**2) / (n - k)
            var_beta = residual_variance * np.linalg.inv(X.T @ X).diagonal()
            se_beta = np.sqrt(var_beta)
            
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'alpha': beta[0],
                'alpha_annual': beta[0] * 12,
                't_stat': t_stats[0],
                'p_value': p_values[0],
                'r_squared': r_squared,
                'n_months': len(merged),
                'betas': {name: beta[i+1] for i, name in enumerate(factor_names)},
                'mean_monthly_return': merged['return'].mean(),
                'volatility': merged['return'].std(),
                'sharpe_ratio': merged['excess_return'].mean() / merged['excess_return'].std() * np.sqrt(12) if merged['excess_return'].std() > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Regression failed: {e}")
            import traceback
            traceback.print_exc()
            return {'alpha': 0.0, 'alpha_annual': 0.0, 't_stat': 0.0, 'p_value': 1.0, 'n_months': len(merged)}
    
    def calculate_risk_metrics(self, monthly_returns: pd.DataFrame, benchmark_returns: pd.Series = None) -> Dict:
        """
        Calculate comprehensive risk-adjusted performance metrics
        
        Following institutional standards for portfolio evaluation:
        - Sharpe Ratio: Treynor & Black (1973)
        - Information Ratio: Grinold & Kahn (2000)
        - Maximum Drawdown: Industry standard
        - Sortino Ratio: Sortino & van der Meer (1991)
        - Calmar Ratio: Young (1991)
        """
        returns = monthly_returns['return'].values
        
        # Annualization factor
        months_per_year = 12
        sqrt_months = np.sqrt(months_per_year)
        
        # Basic statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
        
        # Risk-free rate (2% annual)
        risk_free_monthly = 0.02 / 12
        excess_returns = returns - risk_free_monthly
        mean_excess = np.mean(excess_returns)
        
        # Sharpe Ratio
        sharpe_ratio = (mean_excess / volatility) * sqrt_months if volatility > 0 else 0.0
        
        # Information Ratio
        if benchmark_returns is not None:
            tracking_error = np.std(returns - benchmark_returns, ddof=1)
            active_return = np.mean(returns - benchmark_returns)
            information_ratio = (active_return / tracking_error) * sqrt_months if tracking_error > 0 else 0.0
        else:
            # Use market proxy (risk-free plus 6% equity premium)
            market_return_monthly = risk_free_monthly + (0.06 / 12)
            tracking_error = np.std(returns - market_return_monthly, ddof=1)
            active_return = mean_return - market_return_monthly
            information_ratio = (active_return / tracking_error) * sqrt_months if tracking_error > 0 else 0.0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Find the drawdown period
        max_dd_idx = np.argmin(drawdowns)
        peak_idx = np.argmax(running_max[:max_dd_idx+1]) if max_dd_idx > 0 else 0
        
        # Sortino Ratio (only penalizes downside volatility)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_excess / downside_deviation) * sqrt_months if downside_deviation > 0 else 0.0
        
        # Calmar Ratio
        annualized_return = mean_return * months_per_year
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0.0
        
        # Additional metrics
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        # Distribution shape
        from scipy import stats
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        metrics = {
            # Core risk-adjusted metrics
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Supporting statistics
            'annualized_return': annualized_return,
            'annualized_volatility': volatility * sqrt_months,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            
            # Drawdown details
            'max_drawdown_start': monthly_returns.iloc[peak_idx]['date'] if peak_idx < len(monthly_returns) else None,
            'max_drawdown_end': monthly_returns.iloc[max_dd_idx]['date'] if max_dd_idx < len(monthly_returns) else None,
            'max_drawdown_length_months': max_dd_idx - peak_idx if max_dd_idx > peak_idx else 0,
        }
        
        logger.info(f"Risk metrics calculated:")
        logger.info(f"  Sharpe: {sharpe_ratio:.3f}, Info: {information_ratio:.3f}, MaxDD: {max_drawdown*100:.2f}%")
        
        return metrics
    
    def analyze_portfolio(self, portfolio_name: str, trades_subset: pd.DataFrame, weight_method: str = 'equal') -> Dict:
        """
        Complete analysis for one portfolio, including risk metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING: {portfolio_name}")
        logger.info(f"{'='*80}")
        
        # Build portfolio and calculate returns
        daily_portfolio = self.build_calendar_time_portfolio_vectorized(trades_subset, weight_method)
        monthly_returns = self.calculate_monthly_returns(daily_portfolio)
        
        # Run factor regressions
        results = {}
        for model in ['capm', 'ff3']:
            model_results = self.run_factor_regression(monthly_returns, model=model)
            results[model] = model_results
            
            logger.info(f"\n{model.upper()}:")
            logger.info(f"  Alpha: {model_results['alpha']*100:.2f} bps/month ({model_results['alpha_annual']*100:.2f}% annual)")
            logger.info(f"  t-stat: {model_results['t_stat']:.3f}, p={model_results['p_value']:.4f}")
            if model_results['p_value'] < 0.05:
                logger.info("  Significant at 5% level")
        
        # Calculate risk-adjusted metrics
        risk_metrics = self.calculate_risk_metrics(monthly_returns)
        results['risk_metrics'] = risk_metrics
        
        # Store portfolio data
        results['portfolio_name'] = portfolio_name
        results['n_trades'] = len(trades_subset)
        results['monthly_returns'] = monthly_returns
        results['daily_portfolio'] = daily_portfolio
        
        return results


def run_analysis(price_data_file: Optional[str] = None):
    """Main analysis function"""
    
    print("\n1. Loading data...")
    df = pd.read_pickle('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_relevance_v2_quality.pkl')
    original_count = len(df)
    print(f"Original trades: {original_count:,}")
    print(f"Date range: {df['Traded'].min().date()} to {df['Traded'].max().date()}")
    
    # Check for price data file
    if price_data_file is None:
        possible_paths = [
            '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw/russell_3000_daily.parquet',
            '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/russell_3000_daily.parquet'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                price_data_file = path
                break
    
    if price_data_file:
        print(f"Using price data: {price_data_file}")
    else:
        print("No price data file found")
    
    # Filter trades to match price data availability
    price_data_start = pd.to_datetime('2013-07-31')
    df = df[df['Traded'] >= price_data_start]
    
    print(f"\n After price data filter (>={price_data_start.date()}):")
    print(f"Filtered trades: {len(df):,}")
    print(f"Removed: {original_count - len(df):,} trades before price data availability")
    print(f"Analysis period: {df['Traded'].min().date()} to {df['Traded'].max().date()}")
    print(f"   Duration: {(df['Traded'].max() - df['Traded'].min()).days / 365.25:.1f} years")
    
    # 6-month holding period
    analyzer = CalendarTimePortfolioV2(df, holding_period_days=126, price_data_file=price_data_file)
    
    print("\n2. Loading factors...")
    analyzer.load_factor_data()
    
    print("\n3. Defining portfolios...")
    
    portfolios = {
        # Original threshold
        'High Relevance (>=0.70)': df[(df['relevance_score'] >= 0.70) & (df['direction'] == 1)],
        
        # Tighter threshold
        'High Relevance (>=0.75)': df[(df['relevance_score'] >= 0.75) & (df['direction'] == 1)],
        
        # Senate elite subset
        'Senate Elite (>=0.75)': df[(df['Chamber'] == 'Senate') & 
                                    (df['relevance_score'] >= 0.75) & 
                                    (df['direction'] == 1)],
        
        # Baseline comparisons
        'Low Relevance (<0.3)': df[(df['relevance_score'] < 0.3) & (df['direction'] == 1)],
        'Senate Buys': df[(df['Chamber'] == 'Senate') & (df['direction'] == 1)],
        'House Buys': df[(df['Chamber'] == 'House') & (df['direction'] == 1)],
        'All Buys': df[df['direction'] == 1],
    }
    
    for name, subset in portfolios.items():
        print(f"{name}: {len(subset):,} trades")
    
    print("\n4. Running analysis...")
    
    results = {}
    
    for portfolio_name, trades_subset in portfolios.items():
        if len(trades_subset) < 100:
            print(f"\n Skipping {portfolio_name}: only {len(trades_subset)} trades")
            continue
        
        try:
            result = analyzer.analyze_portfolio(
                portfolio_name, 
                trades_subset,
                weight_method='equal'
            )
            results[portfolio_name] = result
        except Exception as e:
            logger.error(f"Error analyzing {portfolio_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary of results
    summary = []
    for name, result in results.items():
        capm = result.get('capm', {})
        risk_metrics = result.get('risk_metrics', {})
        
        summary.append({
            'Portfolio': name,
            'Trades': result['n_trades'],
            'Months': capm.get('n_months', 0),
            'Alpha (annual)': f"{capm.get('alpha_annual', 0)*100:.2f}%",
            't-stat': f"{capm.get('t_stat', 0):.2f}",
            'Sharpe': f"{risk_metrics.get('sharpe_ratio', 0):.3f}",
            'Info Ratio': f"{risk_metrics.get('information_ratio', 0):.2f}",
            'Max DD': f"{risk_metrics.get('max_drawdown', 0)*100:.1f}%",
            'Sortino': f"{risk_metrics.get('sortino_ratio', 0):.2f}",
            'Win Rate': f"{risk_metrics.get('win_rate', 0)*100:.0f}%",
            'Sig': '***' if capm.get('p_value', 1) < 0.01 else ('**' if capm.get('p_value', 1) < 0.05 else ('*' if capm.get('p_value', 1) < 0.10 else ''))
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Compare different configurations
    if 'High Relevance (>=0.70)' in results and 'High Relevance (>=0.75)' in results:
        hr70 = results['High Relevance (>=0.70)']
        hr75 = results['High Relevance (>=0.75)']
        
        sharpe_70 = hr70['risk_metrics']['sharpe_ratio']
        sharpe_75 = hr75['risk_metrics']['sharpe_ratio']
        sharpe_improvement = ((sharpe_75 - sharpe_70) / sharpe_70) * 100 if sharpe_70 != 0 else 0
        
        print(f"\nThreshold Optimization:")
        print(f"Relevance >=0.70: Sharpe = {sharpe_70:.3f}")
        print(f"Relevance >=0.75: Sharpe = {sharpe_75:.3f}")
        print(f"Improvement: {sharpe_improvement:+.1f}%")
    
    if 'Senate Elite (>=0.75)' in results:
        elite = results['Senate Elite (>=0.75)']
        elite_sharpe = elite['risk_metrics']['sharpe_ratio']
        elite_alpha = elite['capm']['alpha_annual'] * 100
        
        print(f"\nSenate Elite Portfolio:")
        print(f"Sharpe Ratio: {elite_sharpe:.3f}")
        print(f"Annual Alpha: {elite_alpha:.2f}%")
        print(f"Trades: {elite['n_trades']:,}")
    
    detailed_metrics = []
    for name, result in results.items():
        risk = result.get('risk_metrics', {})
        
        detailed_metrics.append({
            'Portfolio': name,
            'Ann. Return': f"{risk.get('annualized_return', 0)*100:.2f}%",
            'Ann. Vol': f"{risk.get('annualized_volatility', 0)*100:.2f}%",
            'Sharpe': f"{risk.get('sharpe_ratio', 0):.3f}",
            'Sortino': f"{risk.get('sortino_ratio', 0):.3f}",
            'Calmar': f"{risk.get('calmar_ratio', 0):.3f}",
            'Max DD': f"{risk.get('max_drawdown', 0)*100:.2f}%",
            'Win Rate': f"{risk.get('win_rate', 0)*100:.1f}%",
            'Win/Loss': f"{risk.get('win_loss_ratio', 0):.2f}"
        })
    
    detailed_df = pd.DataFrame(detailed_metrics)
    print("\n" + detailed_df.to_string(index=False))
    
    # find best portfolio
    best_sharpe = 0
    best_portfolio = None
    for name, result in results.items():
        sharpe = result.get('risk_metrics', {}).get('sharpe_ratio', 0)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_portfolio = name
    
    os.makedirs('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed', exist_ok=True)
    
    # save comprehensive results
    with open('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/sprint_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # save summary tables
    summary_df.to_csv('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/sprint_summary.csv', index=False)
    detailed_df.to_csv('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/sprint_detailed.csv', index=False)
    
    # save risk metrics
    risk_metrics_output = []
    for name, result in results.items():
        risk = result.get('risk_metrics', {})
        capm = result.get('capm', {})
        
        risk_metrics_output.append({
            'portfolio': name,
            'n_trades': result['n_trades'],
            'n_months': capm.get('n_months', 0),
            'alpha_annual': capm.get('alpha_annual', 0),
            't_stat': capm.get('t_stat', 0),
            'p_value': capm.get('p_value', 1),
            **risk
        })
    
    risk_metrics_df = pd.DataFrame(risk_metrics_output)
    risk_metrics_csv = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/sprint_risk_metrics.csv'
    risk_metrics_df.to_csv(risk_metrics_csv, index=False)
    
    return results


if __name__ == "__main__":
    import sys
    
    price_file = None
    if len(sys.argv) > 1:
        price_file = sys.argv[1]
    
    results = run_analysis(price_data_file=price_file)