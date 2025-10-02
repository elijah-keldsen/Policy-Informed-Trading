#!/usr/bin/env python3
"""
Integration Script: Apply Weak Supervision Dataset
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from weak_supervision_labels import WeakSupervisionLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_weak_supervision_to_dataset(
    input_file: str = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_relevance_v2_quality.pkl',
    output_prefix: str = 'trades_with_ws_labels'
):
    """
    Weak supervision framework to dataset
    """
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_pickle(input_file)
        logger.info(f" Loaded {len(df):,} trades from {input_file}")
    except Exception as e:
        logger.error(f"Could not load file: {e}")
        sys.exit(1)
    
    logger.info(f"  Date range: {df['Traded'].min().date()} to {df['Traded'].max().date()}")
    logger.info(f"  Unique politicians: {df['Name'].nunique()}")
    logger.info(f"  Unique tickers: {df['Ticker'].nunique()}")
    
    # Step 2: Check required columns
    print("\n2. Checking required columns...")
    required_cols = ['relevance_score', 'primary_leadership', 'Chamber', 'seniority_years']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error("Cannot proceed without these columns.")
        sys.exit(1)
    
    logger.info(f"All required columns present")
    
    # Check optional columns
    optional_cols = ['days_to_earnings', 'days_to_legislative_action', 
                     'trade_size_percentile', 'subsequent_6m_return']
    available_optional = [col for col in optional_cols if col in df.columns]
    missing_optional = [col for col in optional_cols if col not in df.columns]
    
    logger.info(f"  Available optional columns: {available_optional}")
    if missing_optional:
        logger.info(f"  Missing optional columns (will skip): {missing_optional}")
    
    # Step 3: Initialize weak supervision
    print("\n3. Initializing Weak Supervision Labeler...")
    
    # Check if we have return data for validation
    has_returns = 'subsequent_6m_return' in df.columns
    
    labeler = WeakSupervisionLabeler(
        relevance_high_threshold=0.75,
        relevance_low_threshold=0.30,
        seniority_threshold=16.0,
        use_return_signal=has_returns  # Only if available
    )
    logger.info("Labeler initialized")
    
    # Step 4: Apply weak supervision
    print("\n4. Applying weak supervision (this may take 1-3 minutes)...")
    df_labeled = labeler.label_dataset(df)
    
    # Step 5: Validation (if returns available)
    if has_returns:
        print("\n5. Validating labels against actual returns...")
        labeler.create_comparison_report(df_labeled, return_column='subsequent_6m_return')
    else:
        print("\n5. Skipping validation (no return data available)")
    
    # Step 6: Analyze by portfolio groups
    print("\n6. Analyzing by portfolio groups...")
    
    portfolios = {
        'High Relevance (≥0.75)': df_labeled[df_labeled['relevance_score'] >= 0.75],
        'Low Relevance (<0.3)': df_labeled[df_labeled['relevance_score'] < 0.30],
        'Senate Buys': df_labeled[df_labeled['Chamber'] == 'Senate'],
        'All Buys': df_labeled
    }
    
    for name, subset in portfolios.items():
        labeled_count = subset['ws_label'].notna().sum()
        informed_count = (subset['ws_label'] == 1).sum()
        
        print(f"\n{name}:")
        print(f"  Total trades: {len(subset):,}")
        print(f"  Labeled: {labeled_count:,} ({labeled_count/len(subset)*100:.1f}%)")
        if labeled_count > 0:
            print(f"  Informed: {informed_count:,} ({informed_count/labeled_count*100:.1f}% of labeled)")
        
        # show mean confidence
        if labeled_count > 0:
            mean_conf = subset[subset['ws_label'].notna()]['ws_confidence'].mean()
            print(f"  Mean confidence: {mean_conf:.3f}")
    
    # Step 7: Create filtered datasets
    print("\n7. Creating filtered datasets...")
    
    # high confidence informed trades
    high_conf_informed = df_labeled[
        (df_labeled['ws_label'] == 1) & 
        (df_labeled['ws_confidence'] >= 0.8)
    ]
    
    logger.info(f"  High confidence informed trades: {len(high_conf_informed):,}")
    
    # Step 8: Save results
    print("\n8. Saving results...")
    
    output_dir = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # save full labeled dataset
    output_full = f'{output_dir}/{output_prefix}.pkl'
    df_labeled.to_pickle(output_full)
    logger.info(f"Saved full dataset: {output_full}")
    
    # save high confidence subset
    output_hc = f'{output_dir}/{output_prefix}_high_confidence.pkl'
    high_conf_informed.to_pickle(output_hc)
    logger.info(f"Saved high confidence subset: {output_hc}")
    
    # save CSV for inspection
    output_csv = f'{output_dir}/{output_prefix}.csv'
    df_labeled[['Ticker', 'Traded', 'Name', 'Chamber', 'relevance_score', 
                'primary_leadership', 'seniority_years', 
                'ws_label', 'ws_confidence', 'ws_num_signals']].to_csv(output_csv, index=False)
    logger.info(f"Saved CSV for inspection: {output_csv}")
    
    # Step 9: Create summary report
    print("\n9. Creating summary report...")
    
    summary = {
        'total_trades': len(df_labeled),
        'labeled_trades': df_labeled['ws_label'].notna().sum(),
        'informed_trades': (df_labeled['ws_label'] == 1).sum(),
        'uninformed_trades': (df_labeled['ws_label'] == 0).sum(),
        'mean_confidence': df_labeled[df_labeled['ws_label'].notna()]['ws_confidence'].mean(),
        'high_confidence_informed': len(high_conf_informed),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = f'{output_dir}/{output_prefix}_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary: {summary_file}")
    
    # final summary
    print("Weak Supervision finished")
    print(f"\nResults saved to:")
    print(f"  Full dataset: {output_full}")
    print(f"  High confidence: {output_hc}")
    print(f"  CSV export: {output_csv}")
    print(f"  Summary: {summary_file}")
    
    print(f"\nKey Stats:")
    print(f"  Coverage: {summary['labeled_trades']:,} / {summary['total_trades']:,} ({summary['labeled_trades']/summary['total_trades']*100:.1f}%)")
    print(f"  Informed: {summary['informed_trades']:,} ({summary['informed_trades']/summary['labeled_trades']*100:.1f}% of labeled)")
    print(f"  High confidence: {summary['high_confidence_informed']:,}")
    
    return df_labeled, high_conf_informed, summary


def create_presentation_summary(df_labeled: pd.DataFrame):
    """
    Create key statistics for presentation slides
    """
    
    total = len(df_labeled)
    labeled = df_labeled['ws_label'].notna().sum()
    informed = (df_labeled['ws_label'] == 1).sum()
    
    print(f"\n Coverage:")
    print(f"   {labeled/total*100:.1f}% of trades labeled using weak supervision")
    
    print(f"\n Identification:")
    print(f"   {informed:,} informed trades identified ({informed/labeled*100:.1f}% of labeled)")
    
    print(f"\n Confidence:")
    mean_conf = df_labeled[df_labeled['ws_label'].notna()]['ws_confidence'].mean()
    print(f"   {mean_conf:.2f} average confidence score")
    
    high_conf = (df_labeled[df_labeled['ws_label'].notna()]['ws_confidence'] >= 0.8).sum()
    print(f"   {high_conf:,} high-confidence labels (≥0.8)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run the main pipeline
    df_labeled, high_conf, summary = apply_weak_supervision_to_dataset()
    
    # Create presentation summary
    create_presentation_summary(df_labeled)
    
    print("\n Complete.")