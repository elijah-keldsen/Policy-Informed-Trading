#!/usr/bin/env python3
"""
Cumulative Performance Visualization.
Focused on 2 portfolios + S&P 500 baseline.

Author: ShaneStreet
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf
from scipy import stats as scipy_stats

# B&W styling
plt.style.use('seaborn-v0_8-whitegrid')

# Grayscale only
COLORS = {
    'benchmark': '#808080',      # Medium gray - baseline
    'low_relevance': '#4D4D4D',  # Dark gray - loser
    'high_relevance': '#000000',  # Black - winner
    'grid': '#E0E0E0',
    'positive': '#000000',
    'negative': '#4D4D4D'
}

# Figure settings
DPI = 300
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL = (12, 8)


class PerformanceVisualizer:
    """
    Portfolio performance charts
    Focused on high vs how relevance with S&P 500 baseline
    """
    
    def __init__(self, results: Dict, output_dir: str = None):
        self.results = results
        
        if output_dir is None:
            self.output_dir = Path('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/figures')
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define our two portfolios
        self.portfolio_config = {
            'Low Relevance (<0.3)': {
                'color': COLORS['low_relevance'],
                'linewidth': 2.5,
                'linestyle': '--',
                'alpha': 1.0,
                'label': 'Low Relevance (< 0.3)',
                'order': 1
            },
            'High Relevance (>=0.75)': {
                'color': COLORS['high_relevance'],
                'linewidth': 2.5,
                'linestyle': '-',
                'alpha': 1.0,
                'label': 'High Relevance (≥ 0.75)',
                'order': 2
            }
        }
        
        # Load actual benchmark data
        self.benchmark_returns = None
        self._load_benchmark_data()
        
        print(f"Performance Visualizer initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Focus portfolios: {list(self.portfolio_config.keys())}")
        print(f"Baseline: S&P 500 (actual data)")
    
    def _load_benchmark_data(self):
        """
        Load ACTUAL S&P 500 returns from data
        Priority: 1) Russell 3000 data, 2) Download SPY, 3) Factor data from results
        """
        print("\nLoading actual benchmark data...")
        
        # Get date range from portfolios first
        portfolio_start = None
        portfolio_end = None
        
        for portfolio_name in self.portfolio_config.keys():
            if portfolio_name in self.results:
                result = self.results[portfolio_name]
                monthly_returns = result.get('monthly_returns')
                if monthly_returns is not None:
                    dates = pd.to_datetime(monthly_returns['date'])
                    if portfolio_start is None or dates.min() < portfolio_start:
                        portfolio_start = dates.min()
                    if portfolio_end is None or dates.max() > portfolio_end:
                        portfolio_end = dates.max()
        
        if portfolio_start is None or portfolio_end is None:
            print("Warning: Could not determine portfolio date range")
            portfolio_start = pd.Timestamp('2010-01-01')
            portfolio_end = pd.Timestamp('2025-01-01')
        
        print(f"Portfolio date range: {portfolio_start.date()} to {portfolio_end.date()}")
        
        # Option 1: Try to use Russell 3000 data
        russell_file = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/russell3000_monthly_returns.pkl'
        if os.path.exists(russell_file):
            try:
                russell_data = pd.read_pickle(russell_file)
                if 'SPY' in russell_data.columns or 'SP500' in russell_data.columns:
                    col = 'SPY' if 'SPY' in russell_data.columns else 'SP500'
                    russell_data.index = pd.to_datetime(russell_data.index).tz_localize(None)
                    
                    # Filter to portfolio date range
                    mask = (russell_data.index >= portfolio_start) & (russell_data.index <= portfolio_end)
                    russell_data = russell_data[mask]
                    
                    self.benchmark_returns = pd.DataFrame({
                        'date': russell_data.index,
                        'return': russell_data[col].values
                    })
                    print(f"Loaded S&P 500 returns from Russell 3000 data")
                    print(f"Benchmark months: {len(self.benchmark_returns)}")
                    return
            except Exception as e:
                print(f"Could not load Russell data: {e}")
        
        # Option 2: Try downloading SPY data
        try:
            print("Attempting to download SPY data from Yahoo Finance...")
            spy = yf.Ticker("SPY")
            
            # Add buffer to date range
            start_date = (portfolio_start - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
            end_date = (portfolio_end + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
            
            spy_data = spy.history(start=start_date, end=end_date, interval='1mo')
            
            if len(spy_data) > 0:
                spy_data['return'] = spy_data['Close'].pct_change()
                # Remove timezone information
                spy_data.index = pd.to_datetime(spy_data.index).tz_localize(None)
                
                # Filter to exact portfolio date range
                mask = (spy_data.index >= portfolio_start) & (spy_data.index <= portfolio_end)
                spy_data = spy_data[mask]
                
                self.benchmark_returns = pd.DataFrame({
                    'date': spy_data.index,
                    'return': spy_data['return'].values
                }).dropna()
                
                print(f"Downloaded S&P 500 returns from Yahoo Finance")
                print(f"Benchmark months: {len(self.benchmark_returns)}")
                print(f"Date range: {self.benchmark_returns['date'].min().date()} to {self.benchmark_returns['date'].max().date()}")
                return
        except Exception as e:
            print(f"Could not download SPY: {e}")
    
    def get_portfolio_style(self, portfolio_name):
        """Get styling for a specific portfolio"""
        return self.portfolio_config.get(portfolio_name, {
            'color': None,
            'linewidth': 2.0,
            'linestyle': '-',
            'alpha': 0.8,
            'label': portfolio_name,
            'order': 0
        })
    
    def create_cumulative_returns_chart(self, initial_investment: float = 100000):
        """
        Cumulative returns chart

        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Set white background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        print("\nCreating cumulative returns chart (THE HERO CHART)...")
        
        # Store final values for annotation
        final_values = {}
        
        # Plot S&P 500 baseline
        if self.benchmark_returns is not None:
            dates_benchmark = pd.to_datetime(self.benchmark_returns['date']).dt.tz_localize(None)
            returns_benchmark = self.benchmark_returns['return'].values
            
            benchmark_values = initial_investment * np.cumprod(1 + returns_benchmark)
            
            ax.plot(dates_benchmark, benchmark_values,
                   label='S&P 500 Benchmark',
                   color=COLORS['benchmark'],
                   linestyle=':',
                   linewidth=2.0,
                   alpha=0.9)
            
            final_values['S&P 500'] = {
                'value': benchmark_values[-1],
                'date': dates_benchmark.iloc[-1],
                'color': COLORS['benchmark'],
                'returns': returns_benchmark
            }
        
        # Plot our two portfolios
        portfolio_names = sorted(self.portfolio_config.keys(), 
                                key=lambda x: self.portfolio_config[x]['order'])
        
        for portfolio_name in portfolio_names:
            if portfolio_name not in self.results:
                print(f"Warning: {portfolio_name} not found in results")
                continue
            
            result = self.results[portfolio_name]
            monthly_returns = result.get('monthly_returns')
            
            if monthly_returns is None or len(monthly_returns) == 0:
                print(f"Warning: No monthly returns for {portfolio_name}")
                continue
            
            returns = monthly_returns['return'].values
            dates = monthly_returns['date'].values
            
            cumulative_wealth = initial_investment * np.cumprod(1 + returns)
            style = self.get_portfolio_style(portfolio_name)
            
            ax.plot(dates, cumulative_wealth, 
                   label=style['label'],
                   color=style['color'],
                   linewidth=style['linewidth'],
                   linestyle=style['linestyle'],
                   alpha=style['alpha'])
            
            final_values[portfolio_name] = {
                'value': cumulative_wealth[-1],
                'date': dates[-1],
                'color': style['color'],
                'returns': returns
            }
        
        # Formatting
        ax.set_xlabel('Date', fontsize=11, fontweight='normal', family='serif')
        ax.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='normal', family='serif')
        ax.set_title(f'Smart Filtering Beats the Market: ${initial_investment:,.0f} Investment Growth', 
                    fontsize=13, fontweight='bold', family='serif', pad=15)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Grid - subtle and minimal
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax.axhline(y=initial_investment, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        
        # Legend - clean and minimal
        legend = ax.legend(loc='upper left', frameon=True, shadow=False, fontsize=10,
                          framealpha=1.0, edgecolor='black', fancybox=False)
        legend.get_frame().set_linewidth(0.5)
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=9, colors='black', width=0.8)
        
        # Annotate final values
        annotation_y_offset = 0
        for name in ['S&P 500', 'Low Relevance (<0.3)', 'High Relevance (>=0.75)']:
            if name not in final_values:
                continue
            
            info = final_values[name]
            final_value = info['value']
            total_return = (final_value / initial_investment - 1) * 100
            
            ann_text = f'${final_value/1000:.0f}K\n({total_return:+.0f}%)'
            
            ax.annotate(ann_text,
                       xy=(info['date'], final_value),
                       xytext=(15, annotation_y_offset),
                       textcoords='offset points',
                       fontsize=9,
                       fontweight='normal',
                       family='serif',
                       color=info['color'],
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                edgecolor=info['color'],
                                alpha=1.0,
                                linewidth=1.0))
            
            annotation_y_offset += 30
        
        # Add key insight text box
        high_rel_value = final_values.get('High Relevance (>=0.75)', {}).get('value', 0)
        low_rel_value = final_values.get('Low Relevance (<0.3)', {}).get('value', 0)
        sp500_value = final_values.get('S&P 500', {}).get('value', 0)
        
        if high_rel_value > 0 and sp500_value > 0:
            outperformance_vs_sp500 = ((high_rel_value / sp500_value) - 1) * 100
            underperformance_low = ((low_rel_value / sp500_value) - 1) * 100
            
            insight_text = f'High Relevance:\n+{outperformance_vs_sp500:.0f}% vs S&P 500\n\nLow Relevance:\n{underperformance_low:.0f}% vs S&P 500'
            
            ax.text(0.98, 0.02, insight_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   family='serif',
                   bbox=dict(boxstyle='round,pad=0.8',
                            facecolor='white',
                            edgecolor='black',
                            alpha=1.0,
                            linewidth=0.8))
        
        plt.tight_layout()
        
        filename = self.output_dir / 'hero_chart_cumulative_returns.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        
        plt.close()
    
    def create_side_by_side_comparison(self):
        """
        Side-by-side cumulative returns and drawdown
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Set white background
        fig.patch.set_facecolor('white')
        for ax in [ax1, ax2]:
            ax.set_facecolor('white')
        
        print("\nCreating side-by-side comparison...")
        
        # LEFT: Cumulative returns
        if self.benchmark_returns is not None:
            dates_benchmark = pd.to_datetime(self.benchmark_returns['date']).dt.tz_localize(None)
            returns_benchmark = self.benchmark_returns['return'].values
            cumulative_benchmark = (1 + returns_benchmark).cumprod() - 1
            
            ax1.plot(dates_benchmark, cumulative_benchmark * 100,
                    label='S&P 500',
                    color=COLORS['benchmark'],
                    linestyle=':',
                    linewidth=2.0,
                    alpha=0.9)
        
        for portfolio_name in self.portfolio_config.keys():
            if portfolio_name not in self.results:
                continue
            
            result = self.results[portfolio_name]
            monthly_returns = result.get('monthly_returns')
            
            if monthly_returns is None:
                continue
            
            returns = monthly_returns['return'].values
            dates = pd.to_datetime(monthly_returns['date']).dt.tz_localize(None)
            cumulative_returns = (1 + returns).cumprod() - 1
            
            config = self.get_portfolio_style(portfolio_name)
            
            ax1.plot(dates, cumulative_returns * 100,
                    label=config['label'],
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=config['linewidth'],
                    alpha=1.0)
        
        # Format left plot - cumulative returns
        ax1.set_xlabel('Date', fontsize=11, fontweight='normal', family='serif')
        ax1.set_ylabel('Cumulative Return (%)', fontsize=11, fontweight='normal', family='serif')
        ax1.set_title('Cumulative Returns', fontsize=13, fontweight='bold', family='serif', pad=15)
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        legend1 = ax1.legend(loc='upper left', frameon=True, shadow=False, fontsize=10,
                            framealpha=1.0, edgecolor='black', fancybox=False)
        legend1.get_frame().set_linewidth(0.5)
        
        # RIGHT: Drawdown
        for portfolio_name in self.portfolio_config.keys():
            if portfolio_name not in self.results:
                continue
            
            result = self.results[portfolio_name]
            monthly_returns = result.get('monthly_returns')
            
            if monthly_returns is None:
                continue
            
            returns = monthly_returns['return'].values
            dates = pd.to_datetime(monthly_returns['date']).dt.tz_localize(None)
            
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            config = self.get_portfolio_style(portfolio_name)
            
            ax2.plot(dates, drawdown * 100,
                    label=config['label'],
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=config['linewidth'],
                    alpha=1.0)
        
        # Plot benchmark drawdown
        if self.benchmark_returns is not None:
            cumulative_benchmark = (1 + returns_benchmark).cumprod()
            running_max_benchmark = np.maximum.accumulate(cumulative_benchmark)
            drawdown_benchmark = (cumulative_benchmark - running_max_benchmark) / running_max_benchmark
            
            ax2.plot(dates_benchmark, drawdown_benchmark * 100,
                    label='S&P 500',
                    color=COLORS['benchmark'],
                    linestyle=':',
                    linewidth=2.0,
                    alpha=0.9)
        
        # Format right plot - drawdown
        ax2.set_xlabel('Date', fontsize=11, fontweight='normal', family='serif')
        ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='normal', family='serif')
        ax2.set_title('Maximum Drawdown', fontsize=13, fontweight='bold', family='serif', pad=15)
        ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        legend2 = ax2.legend(loc='lower left', frameon=True, shadow=False, fontsize=10,
                            framealpha=1.0, edgecolor='black', fancybox=False)
        legend2.get_frame().set_linewidth(0.5)
        
        # Clean spines for both plots
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.tick_params(axis='both', which='major', labelsize=9, colors='black', width=0.8)
        
        plt.tight_layout()
        
        filename = self.output_dir / 'drawdown_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        
        plt.close()
    
    def create_rolling_alpha_comparison(self, window_months: int = 12):
        """
        Rolling 12-month alpha comparison
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Set white background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        print("\nCreating rolling alpha comparison...")
        
        portfolio_names = ['Low Relevance (<0.3)', 'High Relevance (>=0.75)']
        
        # Get benchmark returns for comparison
        benchmark_returns_dict = {}
        if self.benchmark_returns is not None:
            for _, row in self.benchmark_returns.iterrows():
                benchmark_returns_dict[pd.Timestamp(row['date'])] = row['return']
        
        for portfolio_name in portfolio_names:
            if portfolio_name not in self.results:
                continue
            
            result = self.results[portfolio_name]
            monthly_returns = result.get('monthly_returns')
            
            if monthly_returns is None or len(monthly_returns) == 0:
                continue
            
            returns = monthly_returns['return'].values
            dates = monthly_returns['date'].values
            
            # Calculate excess returns vs benchmark
            excess_returns = []
            for i, date in enumerate(dates):
                bench_return = benchmark_returns_dict.get(pd.Timestamp(date), 0.01)
                excess_returns.append(returns[i] - bench_return)
            
            excess_returns = np.array(excess_returns)
            
            if len(returns) < window_months:
                continue
            
            rolling_alpha = []
            rolling_dates = []
            
            for i in range(window_months - 1, len(returns)):
                window_returns = excess_returns[i - window_months + 1:i + 1]
                annual_alpha = np.mean(window_returns) * 12 * 100
                rolling_alpha.append(annual_alpha)
                rolling_dates.append(dates[i])
            
            config = self.get_portfolio_style(portfolio_name)
            
            ax.plot(rolling_dates, rolling_alpha,
                   label=config['label'],
                   color=config['color'],
                   linestyle=config['linestyle'],
                   linewidth=config['linewidth'],
                   alpha=1.0)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=11, fontweight='normal', family='serif')
        ax.set_ylabel('12-Month Rolling Alpha (%)', fontsize=11, fontweight='normal', family='serif')
        ax.set_title('Rolling Alpha: Consistency Over Time', fontsize=13, fontweight='bold', family='serif', pad=15)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        
        legend = ax.legend(loc='upper left', frameon=True, shadow=False, fontsize=10,
                          framealpha=1.0, edgecolor='black', fancybox=False)
        legend.get_frame().set_linewidth(0.5)
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='major', labelsize=9, colors='black', width=0.8)
        
        plt.tight_layout()
        
        filename = self.output_dir / 'rolling_alpha_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        
        plt.close()
    
    def create_return_distribution_comparison(self):
        """
        Distribution histograms
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Set white background
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')
        
        print("\nCreating return distributions...")
        
        portfolio_names = ['Low Relevance (<0.3)', 'High Relevance (>=0.75)']
        
        for idx, portfolio_name in enumerate(portfolio_names):
            if portfolio_name not in self.results:
                continue
            
            result = self.results[portfolio_name]
            monthly_returns = result.get('monthly_returns')
            
            if monthly_returns is None:
                continue
            
            returns = monthly_returns['return'].values
            config = self.get_portfolio_style(portfolio_name)
            
            # Plot histogram
            axes[idx].hist(returns * 100, bins=30, 
                          color=config['color'], 
                          alpha=0.6,
                          edgecolor='black',
                          linewidth=0.5)
            
            # Add vertical line at mean
            mean_return = np.mean(returns) * 100
            axes[idx].axvline(mean_return, color=config['color'], 
                            linestyle='-', linewidth=2.0,
                            label=f'Mean: {mean_return:.2f}%')
            
            # Format
            axes[idx].set_xlabel('Monthly Return (%)', fontsize=11, fontweight='normal', family='serif')
            axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='normal', family='serif')
            axes[idx].set_title(config['label'], fontsize=12, fontweight='bold', family='serif', pad=10)
            axes[idx].grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
            
            legend = axes[idx].legend(frameon=True, shadow=False, fontsize=10,
                                     framealpha=1.0, edgecolor='black', fancybox=False)
            legend.get_frame().set_linewidth(0.5)
            
            # Clean spines
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].spines['left'].set_color('black')
            axes[idx].spines['bottom'].set_color('black')
            axes[idx].spines['left'].set_linewidth(0.8)
            axes[idx].spines['bottom'].set_linewidth(0.8)
            axes[idx].tick_params(axis='both', which='major', labelsize=9, colors='black', width=0.8)
        
        plt.suptitle('Return Distributions', fontsize=14, fontweight='bold', family='serif', y=1.02)
        plt.tight_layout()
        
        filename = self.output_dir / 'return_distributions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        
        plt.close()
    
    def create_performance_comparison_table(self):
        """
        Performance metrics comparison table
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Set white background
        fig.patch.set_facecolor('white')
        ax.axis('off')
        
        print("\nCreating performance comparison table...")
        
        portfolio_names = ['Low Relevance (<0.3)', 'High Relevance (>=0.75)']
        
        table_data = []
        
        # Add S&P 500 benchmark
        if self.benchmark_returns is not None and len(portfolio_names) > 0:
            first_portfolio = portfolio_names[0]
            if first_portfolio in self.results:
                result = self.results[first_portfolio]
                monthly_returns = result.get('monthly_returns')
                
                if monthly_returns is not None:
                    # Get portfolio dates and remove timezone
                    portfolio_dates = pd.to_datetime(monthly_returns['date']).dt.tz_localize(None)
                    
                    # Create alignment DataFrame
                    benchmark_aligned = pd.DataFrame({'date': portfolio_dates})
                    
                    # Prepare benchmark data - remove timezone
                    bench_df = self.benchmark_returns.copy()
                    bench_df['date'] = pd.to_datetime(bench_df['date']).dt.tz_localize(None)
                    
                    # Merge
                    benchmark_aligned = benchmark_aligned.merge(bench_df, on='date', how='left')
                    
                    # Fill missing values
                    benchmark_aligned['return'] = benchmark_aligned['return'].ffill()
                    benchmark_aligned['return'] = benchmark_aligned['return'].bfill()
                    
                    if benchmark_aligned['return'].isna().any():
                        benchmark_aligned['return'] = benchmark_aligned['return'].fillna(0.01)
                    
                    bench_returns = benchmark_aligned['return'].values
                    total_return = (np.prod(1 + bench_returns) - 1) * 100
                    n_months = len(bench_returns)
                    annualized_return = ((1 + total_return/100) ** (12/n_months) - 1) * 100
                    
                    volatility = np.std(bench_returns) * np.sqrt(12)
                    risk_free_annual = 2.0
                    sharpe = (annualized_return - risk_free_annual) / (volatility * 100) if volatility > 0 else 0
                    
                    cumulative = np.cumprod(1 + bench_returns)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    max_dd = np.min(drawdown) * 100
                    
                    win_rate = np.sum(bench_returns > 0) / len(bench_returns) * 100
                    
                    table_data.append({
                        'Strategy': 'S&P 500 Baseline',
                        'Total Return': f"{total_return:.1f}%",
                        'Annual Alpha': f"0.00%",
                        'Sharpe Ratio': f"{sharpe:.3f}",
                        'Sortino Ratio': f"-",
                        'Max Drawdown': f"{max_dd:.1f}%",
                        'Win Rate': f"{win_rate:.0f}%",
                        't-stat': f"-",
                        'Sig.': f"-"
                    })
        
        # Add portfolio data
        for portfolio_name in portfolio_names:
            if portfolio_name not in self.results:
                continue
            
            result = self.results[portfolio_name]
            monthly_returns = result.get('monthly_returns')
            
            if monthly_returns is None:
                continue
            
            returns = monthly_returns['return'].values
            
            # Calculate metrics
            total_return = (np.prod(1 + returns) - 1) * 100
            n_months = len(returns)
            annualized_return = ((1 + total_return/100) ** (12/n_months) - 1) * 100
            
            # Sharpe ratio
            volatility = np.std(returns) * np.sqrt(12)
            risk_free_annual = 2.0
            sharpe = (annualized_return - risk_free_annual) / (volatility * 100) if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) * np.sqrt(12) if len(downside_returns) > 0 else volatility
            sortino = (annualized_return - risk_free_annual) / (downside_std * 100) if downside_std > 0 else 0
            
            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown) * 100
            
            # Win rate
            win_rate = np.sum(returns > 0) / len(returns) * 100
            
            # Alpha and t-stat (vs benchmark)
            capm_results = result.get('capm', {})
            alpha_annual = capm_results.get('alpha_annual', 0) * 100
            t_stat = capm_results.get('t_stat', 0)
            p_value = capm_results.get('p_value', 1)
            
            # Significance stars
            if p_value < 0.01:
                sig = "***"
            elif p_value < 0.05:
                sig = "**"
            elif p_value < 0.10:
                sig = "*"
            else:
                sig = ""
            
            table_data.append({
                'Strategy': portfolio_name,
                'Total Return': f"{total_return:.1f}%",
                'Annual Alpha': f"{alpha_annual:.2f}%",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Sortino Ratio': f"{sortino:.3f}",
                'Max Drawdown': f"{max_dd:.1f}%",
                'Win Rate': f"{win_rate:.0f}%",
                't-stat': f"{t_stat:.2f}",
                'Sig.': sig
            })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Create table
        col_widths = [0.20, 0.12, 0.12, 0.11, 0.11, 0.12, 0.10, 0.08, 0.04]
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', colWidths=col_widths,
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for j in range(len(df.columns)):
            cell = table[(0, j)]
            cell.set_facecolor('#D3D3D3')  # Light gray
            cell.set_text_props(weight='bold', family='serif', size=10)
            cell.set_edgecolor('black')
            cell.set_linewidth(1.0)
        
        # Style data rows
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                cell = table[(i, j)]
                cell.set_facecolor('white')
                cell.set_text_props(family='serif', size=10)
                cell.set_edgecolor('black')
                cell.set_linewidth(0.5)
                
                # Highlight portfolio names
                if j == 0:
                    cell.set_text_props(weight='bold', family='serif', size=10)
                
                # Light highlighting for best values (subtle gray)
                if j > 0 and i > 0:
                    try:
                        cell_value = df.iloc[i-1, j]
                        # Extract numeric value
                        if '%' in str(cell_value):
                            numeric_value = float(str(cell_value).replace('%', '').replace('-', '0'))
                            col_values = [float(str(v).replace('%', '').replace('-', '0')) 
                                        for v in df.iloc[:, j] if v and v not in ['-', '']]
                            if len(col_values) > 0 and numeric_value == max(col_values):
                                cell.set_facecolor('#F0F0F0')  # Very light gray
                    except:
                        pass
        
        plt.title('Performance Comparison: High vs Low Relevance vs Market', 
                fontsize=13, fontweight='bold', family='serif', pad=20)
        
        filename = self.output_dir / 'performance_table.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        
        plt.close()
        
        # Also save as CSV
        csv_filename = self.output_dir / 'performance_table.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations for our High vs Low story
        """
        print("="*80)
        print("\nNarrative: Smart Filtering Beats the Market")
        print("Portfolios:")
        print("1. S&P 500 - Market Baseline")
        print("2. Low Relevance (<0.3) - What NOT to do")
        print("3. High Relevance (>=0.75) - Our Strategy")
        print("="*80)
        
        # Verify we're using actual data
        print("\nDATA VERIFICATION:")
        for portfolio_name in ['Low Relevance (<0.3)', 'High Relevance (>=0.75)']:
            if portfolio_name in self.results:
                result = self.results[portfolio_name]
                monthly_returns = result.get('monthly_returns')
                if monthly_returns is not None:
                    returns = monthly_returns['return'].values
                    print(f"{portfolio_name}:")
                    print(f"N months: {len(returns)}")
                    print(f"Mean return: {np.mean(returns)*100:.3f}%/month")
                    print(f"Cumulative: {(np.prod(1 + returns) - 1)*100:.1f}%")
        
        if self.benchmark_returns is not None:
            print(f"S&P 500 Benchmark:")
            print(f"N months: {len(self.benchmark_returns)}")
            print(f"Mean return: {self.benchmark_returns['return'].mean()*100:.3f}%/month")
        
        # Generate all charts
        self.create_cumulative_returns_chart()
        self.create_side_by_side_comparison()
        self.create_rolling_alpha_comparison()
        self.create_return_distribution_comparison()
        self.create_performance_comparison_table()
        
        print("\n" + "="*80)
        print("COMPLETE")
        print("="*80)
        print(f"\nFiles saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*.png')):
            print(f"  - {file.name}")

def main():
    """
    Load results and generate visualizations
    """
    
    # Load results
    results_file = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/sprint_results.pkl'
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} portfolio results")
    
    # Verify two key portfolios
    required_portfolios = ['Low Relevance (<0.3)', 'High Relevance (>=0.75)']
    missing = [p for p in required_portfolios if p not in results]
    
    if missing:
        print(f"\nError: Missing required portfolios: {missing}")
        print(f"Available portfolios: {list(results.keys())}")
        return
    
    print("\nAll required portfolios located")
    
    # Create visualizer
    visualizer = PerformanceVisualizer(results)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*80)
    print("Visualizations Complete")

if __name__ == "__main__":
    main()