#!/usr/bin/env python3
"""
Apply Committee Relevance Scoring to full dataset
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from committee_mapper_v2 import CommitteeSectorMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_relevance_scoring_v2(input_file: str, output_prefix: str = 'trades_with_relevance_v2'):
    """
    Apply committee relevance scoring to dataset
    """
    
    logger.info(f"1. Loading from: {input_file}")
    
    try:
        df = pd.read_pickle(input_file)
        logger.info(f"Loaded {len(df):,} trades")
    except Exception as e:
        logger.error(f"Could not load file: {e}")
        sys.exit(1)
    
    required_cols = ['Ticker', 'committees_list', 'leadership_roles', 'seniority_years']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    print("\n2. Initializing Committee-Sector Mapper V2...")
    print("   (Enhanced with fuzzy matching and expanded sectors)")
    mapper = CommitteeSectorMapper()
    
    print("\n3. Calculating relevance scores...")
    print("   (This may take a few minutes...)")
    
    df_with_scores = mapper.calculate_relevance_for_dataframe(df)
    
    print("\n4. Analyzing results...")
    
    print(f"\nOverall Statistics:")
    print(f"   Mean score: {df_with_scores['relevance_score'].mean():.3f}")
    print(f"   Median score: {df_with_scores['relevance_score'].median():.3f}")
    print(f"   Std dev: {df_with_scores['relevance_score'].std():.3f}")
    
    print(f"\n Distribution:")
    high = (df_with_scores['relevance_score'] >= 0.7).sum()
    medium = ((df_with_scores['relevance_score'] >= 0.3) & (df_with_scores['relevance_score'] < 0.7)).sum()
    low = (df_with_scores['relevance_score'] < 0.3).sum()
    
    print(f"   High relevance (≥0.7):   {high:>7,} ({high/len(df_with_scores)*100:>5.1f}%)")
    print(f"   Medium relevance (0.3-0.7): {medium:>7,} ({medium/len(df_with_scores)*100:>5.1f}%)")
    print(f"   Low relevance (<0.3):    {low:>7,} ({low/len(df_with_scores)*100:>5.1f}%)")
    
    unknown_committee = (df_with_scores['most_relevant_committee'] == 'Unknown').sum()
    print(f"\n   Unknown committee:       {unknown_committee:>7,} ({unknown_committee/len(df_with_scores)*100:>5.1f}%)")
    
    print(f"\n Top Committees by Trade Count:")
    top_committees = df_with_scores['most_relevant_committee'].value_counts().head(15)
    for committee, count in top_committees.items():
        avg_score = df_with_scores[df_with_scores['most_relevant_committee'] == committee]['relevance_score'].mean()
        pct = count / len(df_with_scores) * 100
        print(f"   {committee[:55]:<55} {count:>6,} ({pct:>4.1f}%) avg: {avg_score:.3f}")
    
    print(f"\n Top Sectors Traded:")
    top_sectors = df_with_scores['sector'].value_counts().head(15)
    for sector, count in top_sectors.items():
        avg_score = df_with_scores[df_with_scores['sector'] == sector]['relevance_score'].mean()
        print(f"   {sector[:55]:<55} {count:>6,} avg: {avg_score:.3f}")
    
    print(f"\n By Chamber:")
    for chamber in ['Senate', 'House']:
        chamber_df = df_with_scores[df_with_scores['Chamber'] == chamber]
        high_rel = (chamber_df['relevance_score'] >= 0.7).sum()
        print(f"   {chamber}:")
        print(f"      Trades: {len(chamber_df):,}")
        print(f"      Avg relevance: {chamber_df['relevance_score'].mean():.3f}")
        print(f"      High relevance (≥0.7): {high_rel:,} ({high_rel/len(chamber_df)*100:.1f}%)")
    
    print(f"\n By Transaction Type:")
    for direction, label in [(1, 'Buys'), (-1, 'Sells')]:
        dir_df = df_with_scores[df_with_scores['direction'] == direction]
        high_rel = (dir_df['relevance_score'] >= 0.7).sum()
        print(f"   {label}:")
        print(f"      Trades: {len(dir_df):,}")
        print(f"      Avg relevance: {dir_df['relevance_score'].mean():.3f}")
        print(f"      High relevance (≥0.7): {high_rel:,} ({high_rel/len(dir_df)*100:.1f}%)")
    
    print(f"\n Examples of high relevance trades (score ≥ 0.9):")
    high_rel = df_with_scores[df_with_scores['relevance_score'] >= 0.9].sort_values('relevance_score', ascending=False).head(10)
    
    if len(high_rel) > 0:
        for idx, row in high_rel.iterrows():
            print(f"   {row['Name'][:25]:<25} {row['Transaction']:<10} {row['Ticker']:<6} "
                  f"{row['sector'][:25]:<25} Score: {row['relevance_score']:.3f}")
    
    # Comparison with V1 if available
    print(f"\n IMPROVEMENT ANALYSIS:")
    print(f"   V2 High relevance trades: {high:,} ({high/len(df_with_scores)*100:.1f}%)")
    print(f"   V2 Unknown committee: {unknown_committee:,} ({unknown_committee/len(df_with_scores)*100:.1f}%)")
    
    print("\n5. Saving results...")
    os.makedirs('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed', exist_ok=True)
    
    output_csv = f'/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/{output_prefix}.csv'
    output_pkl = f'/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/{output_prefix}.pkl'
    output_parquet = f'/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/{output_prefix}.parquet'
    
    # CSV export
    df_csv = df_with_scores.copy()
    list_cols = ['committees_list', 'committee_codes', 'leadership_roles', 'committee_types']
    for col in list_cols:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else str(x)
            )
    
    df_csv.to_csv(output_csv, index=False)
    df_with_scores.to_pickle(output_pkl)
    df_with_scores.to_parquet(output_parquet)
    
    print(f"\n Saved to:")
    print(f"CSV:     {output_csv}")
    print(f"Pickle:  {output_pkl}")
    print(f"Parquet: {output_parquet}")
    
    return df_with_scores

if __name__ == "__main__":
    input_file = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.pkl'
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    df_scored = apply_relevance_scoring_v2(
        input_file=input_file,
        output_prefix='trades_with_relevance_v2_quality'
    )