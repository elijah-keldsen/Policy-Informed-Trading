#!/usr/bin/env python3
"""
Simplified Presentation Visualizations

Author: ShaneStreet
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# B&W styling
plt.style.use('seaborn-v0_8-whitegrid')

# Grayscale only
COLORS = {
    'high_relevance': '#000000',    # Black (solid line)
    'low_relevance': '#4D4D4D',     # Dark gray (dashed line)
    'sp500': '#808080',             # Medium gray (dotted line)
}

SPLIT_DATE = pd.Timestamp('2020-01-01')


class SimpleInOutVisualizer:
    """
    Create B&W visualizations
    """
    
    def __init__(self, results: Dict, output_dir: str = './output'):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparison = results.get('comparison', {})
        self.in_sample = results.get('in_sample', {})
        self.out_sample = results.get('out_sample', {})
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        logger.info("Creating visualizations...")
        
        # 1. Table
        self.create_metrics_table()
        
        # 2-5. Four comparison graphs
        self.create_in_sample_high_vs_sp500()
        self.create_in_sample_low_vs_sp500()
        self.create_out_sample_high_vs_sp500()
        self.create_out_sample_low_vs_sp500()
        
        logger.info(f"\n Visualizations saved to {self.output_dir}")
    
    def create_metrics_table(self):
        """Performance metrics table"""
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis('off')
        
        portfolio_order = ['S&P 500', 'Low Relevance (<0.3)', 'High Relevance (≥0.75)']
        
        table_data = [
            ['Portfolio', 'In-Sample\nSharpe', 'Out-of-Sample\nSharpe', 
             'In-Sample\nAlpha', 'Out-of-Sample\nAlpha',
             'In-Sample\nt-stat', 'Out-of-Sample\nt-stat']
        ]
        
        for portfolio_name in portfolio_order:
            if portfolio_name in self.comparison:
                comp = self.comparison[portfolio_name]
                in_s = comp['in_sample']
                out_s = comp['out_sample']
                
                if portfolio_name == 'S&P 500':
                    in_alpha = "-"
                    out_alpha = "-"
                else:
                    in_alpha = f"{in_s['alpha_annual']*100:.2f}%" if in_s['alpha_annual'] != 0 else "0.00%"
                    out_alpha = f"{out_s['alpha_annual']*100:.2f}%" if out_s['alpha_annual'] != 0 else "0.00%"
                
                row = [
                    portfolio_name,
                    f"{in_s['sharpe']:.3f}",
                    f"{out_s['sharpe']:.3f}",
                    in_alpha,
                    out_alpha,
                    f"{in_s['t_stat']:.2f}" if in_s['t_stat'] != 0 else "-",
                    f"{out_s['t_stat']:.2f}" if out_s['t_stat'] != 0 else "-"
                ]
                table_data.append(row)
        
        col_widths = [0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        
        table = ax.table(cellText=table_data, cellLoc='center',
                        colWidths=col_widths, bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)
        
        # Header - bold black on white
        for j in range(7):
            cell = table[(0, j)]
            cell.set_facecolor('white')
            cell.set_text_props(weight='bold', color='black', size=12, family='serif')
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
        
        # Data rows - clean alternating pattern
        for i in range(1, len(table_data)):
            for j in range(7):
                cell = table[(i, j)]
                
                # Very subtle alternating pattern
                if i % 2 == 1:
                    cell.set_facecolor('#F8F8F8')  # Barely gray
                else:
                    cell.set_facecolor('white')
                
                if j == 0:
                    cell.set_text_props(weight='bold', size=11, color='black', family='serif')
                    cell._loc = 'left'
                else:
                    cell.set_text_props(color='black', size=11, family='serif')
                
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        plt.title('Performance Metrics: In-Sample vs Out-of-Sample',
                 fontsize=14, fontweight='bold', family='serif', pad=15)
        
        output_file = self.output_dir / 'performance_metrics_table.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_file}")
        plt.close()
    
    def _get_filtered_data(self, portfolio_name: str, sample_type: str):
        """Helper to get filtered portfolio data"""
        if sample_type == 'in_sample':
            data = self.comparison[portfolio_name]['in_sample']['monthly_returns']
            date_filter = lambda d: d < SPLIT_DATE
        else:
            data = self.comparison[portfolio_name]['out_sample']['monthly_returns']
            date_filter = lambda d: d >= SPLIT_DATE
        
        if data is None or len(data) == 0:
            return None, None
        
        returns = data['return'].values
        dates = pd.to_datetime(data['date'])
        
        # Create mask as numpy array to avoid index alignment issues
        mask = np.array([date_filter(d) for d in dates])
        
        # Apply mask to get filtered data
        filtered_dates = dates[mask]
        filtered_returns = returns[mask]
        
        # Reset index to avoid issues
        filtered_dates = pd.Series(filtered_dates).reset_index(drop=True)
        
        return filtered_dates, filtered_returns
    
    def _create_graph(self, portfolio_name, sample_type, output_filename, title):
        """Create a single comparison graph with B&W"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # White background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Get portfolio data
        dates_portfolio, returns_portfolio = self._get_filtered_data(portfolio_name, sample_type)
        
        # Determine line style and color
        if 'High' in portfolio_name:
            portfolio_color = COLORS['high_relevance']
            portfolio_linestyle = '-'
            portfolio_linewidth = 2.5
            portfolio_label = 'High Relevance Strategy'
        else:
            portfolio_color = COLORS['low_relevance']
            portfolio_linestyle = '--'
            portfolio_linewidth = 2.5
            portfolio_label = 'Low Relevance Strategy'
        
        # Plot portfolio
        if dates_portfolio is not None:
            cum_log_portfolio = np.cumsum(np.log(1 + returns_portfolio)) * 100
            ax.plot(dates_portfolio, cum_log_portfolio,
                   label=portfolio_label,
                   color=portfolio_color,
                   linestyle=portfolio_linestyle,
                   linewidth=portfolio_linewidth,
                   alpha=1.0)
        
        # Get S&P 500 data and align
        dates_sp500, returns_sp500 = self._get_filtered_data('S&P 500', sample_type)
        
        if dates_sp500 is not None and dates_portfolio is not None:
            df_sp500 = pd.DataFrame({'date': dates_sp500, 'return': returns_sp500})
            df_portfolio = pd.DataFrame({'date': dates_portfolio})
            aligned = df_portfolio.merge(df_sp500, on='date', how='left')
            
            # Pandas methods
            aligned['return'] = aligned['return'].interpolate(method='linear')
            aligned['return'] = aligned['return'].bfill()  # Backward fill
            aligned['return'] = aligned['return'].ffill()  # Forward fill
            
            cum_log_sp500 = np.cumsum(np.log(1 + aligned['return'].values)) * 100
            ax.plot(dates_portfolio, cum_log_sp500,
                   label='S&P 500 Benchmark',
                   color=COLORS['sp500'],
                   linestyle=':',
                   linewidth=2.0,
                   alpha=0.9)
        
        # Styling
        ax.set_xlabel('Date', fontsize=11, fontweight='normal', family='serif')
        ax.set_ylabel('Cumulative Log Return (%)', fontsize=11, fontweight='normal', family='serif')
        ax.set_title(title, fontsize=13, fontweight='bold', family='serif', pad=15)
        
        # Legend
        legend = ax.legend(loc='upper left', frameon=True, shadow=False, fontsize=10, 
                          framealpha=1.0, edgecolor='black', fancybox=False)
        legend.get_frame().set_linewidth(0.5)
        
        # Grid
        ax.grid(True, alpha=0.2, linewidth=0.5, color='gray', linestyle='-')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=9, colors='black', width=0.8)
        
        plt.tight_layout()
        
        output_file = self.output_dir / output_filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_file}")
        plt.close()
    
    def create_in_sample_high_vs_sp500(self):
        """Graph 1: In-Sample High Relevance vs S&P 500"""
        self._create_graph(
            'High Relevance (≥0.75)',
            'in_sample',
            'in_sample_high_vs_sp500.png',
            'In-Sample Performance: High Relevance vs S&P 500 (2013-2019)'
        )
    
    def create_in_sample_low_vs_sp500(self):
        """Graph 2: In-Sample Low Relevance vs S&P 500"""
        self._create_graph(
            'Low Relevance (<0.3)',
            'in_sample',
            'in_sample_low_vs_sp500.png',
            'In-Sample Performance: Low Relevance vs S&P 500 (2013-2019)'
        )
    
    def create_out_sample_high_vs_sp500(self):
        """Graph 3: Out-of-Sample High Relevance vs S&P 500"""
        self._create_graph(
            'High Relevance (≥0.75)',
            'out_sample',
            'out_sample_high_vs_sp500.png',
            'Out-of-Sample Performance: High Relevance vs S&P 500 (2020-2025)'
        )
    
    def create_out_sample_low_vs_sp500(self):
        """Graph 4: Out-of-Sample Low Relevance vs S&P 500"""
        self._create_graph(
            'Low Relevance (<0.3)',
            'out_sample',
            'out_sample_low_vs_sp500.png',
            'Out-of-Sample Performance: Low Relevance vs S&P 500 (2020-2025)'
        )


def main():
    """Generate visualizations"""
    results_file = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/in_out_sample_results_v2.pkl'
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    viz = SimpleInOutVisualizer(results, output_dir='./output')
    viz.create_all_visualizations()

if __name__ == "__main__":
    main()