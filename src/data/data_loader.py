#!/usr/bin/env python3
"""
Congressional Trading Data Loader.
Implements Ziobrowski et al. (2004) screening criteria with modern enhancements

Author: ShaneStreet
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import requests
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CongressionalDataLoader:
    """
    Load and prepare congressional trading data
    Implements Ziobrowski et al. (2004) screening criteria
    """
    
    def __init__(self, trades_file='/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw/congresstradingall.csv'):
        self.trades_file = trades_file
        self.raw_trades = None
        self.clean_trades = None
        
    def load_raw_trades(self):
        """
        Load raw congressional trading data
        """
        logger.info(f"Loading trades from {self.trades_file}")
        
        df = pd.read_csv(self.trades_file)
        
        # Convert dates to datetime
        df['Traded'] = pd.to_datetime(df['Traded'], errors='coerce')
        df['Filed'] = pd.to_datetime(df['Filed'], errors='coerce')
        
        # Remove timezone if present
        if hasattr(df['Traded'].dtype, 'tz') and df['Traded'].dt.tz is not None:
            df['Traded'] = df['Traded'].dt.tz_localize(None)
        if hasattr(df['Filed'].dtype, 'tz') and df['Filed'].dt.tz is not None:
            df['Filed'] = df['Filed'].dt.tz_localize(None)
        
        logger.info(f"Loaded {len(df)} raw trades")
        logger.info(f"Date range: {df['Traded'].min()} to {df['Traded'].max()}")
        
        self.raw_trades = df
        return df
    
    def apply_ziobrowski_screening(self):
        """
        Apply Ziobrowski et al. (2004) screening criteria
        """
        df = self.raw_trades.copy()
        
        initial_count = len(df)
        logger.info(f"Starting with {initial_count} trades")
        
        # Filter 1: Handle TickerType
        if 'TickerType' in df.columns:
            exclude_types = ['Corporate Bond', 'Municipal Security', 'Bond', 
                           'Mutual Fund', 'ETF', 'Options', 'Cryptocurrency',
                           'Money Market', 'Alternative Investment']
            
            df = df[~df['TickerType'].isin(exclude_types)]
            logger.info(f"After excluding non-stock types: {len(df)} ({len(df)/initial_count*100:.1f}%)")
        else:
            logger.info("No TickerType column - skipping this filter")
        
        # Filter 2: Valid ticker
        before = len(df)
        df = df[df['Ticker'].notna()]
        logger.info(f"After removing null tickers: {len(df)} (removed {before-len(df)})")
        
        before = len(df)
        df = df[df['Ticker'].str.len() <= 6]
        logger.info(f"After ticker length filter (≤6 chars): {len(df)} (removed {before-len(df)})")
        
        before = len(df)
        df = df[~df['Ticker'].str.contains(r'^[0-9]', na=False, regex=True)]
        logger.info(f"After removing tickers starting with numbers: {len(df)} (removed {before-len(df)})")
        
        # Filter 3: Valid dates
        before = len(df)
        df = df[df['Traded'].notna() & df['Filed'].notna()]
        logger.info(f"After date validation: {len(df)} (removed {before-len(df)})")
        
        # Filter 4: Post-STOCK Act (2012-04-04 onward)
        before = len(df)
        stock_act_date = pd.to_datetime('2012-04-04')
        df = df[df['Traded'] >= stock_act_date]
        logger.info(f"After STOCK Act filter (≥2012-04-04): {len(df)} (removed {before-len(df)})")
        
        # Filter 5: Valid transaction types
        before = len(df)
        valid_patterns = ['Purchase', 'Sale', 'Sell', 'Buy']
        df = df[df['Transaction'].str.contains('|'.join(valid_patterns), case=False, na=False)]
        logger.info(f"After transaction type filter: {len(df)} (removed {before-len(df)})")
        
        # Filter 6: Disclosure lag validation
        before = len(df)
        df['disclosure_lag_days'] = (df['Filed'] - df['Traded']).dt.days
        df = df[(df['disclosure_lag_days'] >= 0) & (df['disclosure_lag_days'] <= 180)]
        logger.info(f"After disclosure lag filter (0-180 days): {len(df)} (removed {before-len(df)})")
        
        # Filter 7: Remove future trades
        before = len(df)
        df = df[df['Traded'] <= datetime.now()]
        logger.info(f"After future trade filter: {len(df)} (removed {before-len(df)})")
        
        # Parse trade amounts
        logger.info(f"\nParsing trade amounts...")
        df = self._parse_trade_amounts(df)
        
        # Create transaction direction
        df['direction'] = df['Transaction'].apply(
            lambda x: 1 if 'Purchase' in str(x) or 'Buy' in str(x) else -1
        )
        
        logger.info(f"Final clean dataset: {len(df)} trades ({len(df)/initial_count*100:.1f}% of initial)")
        logger.info(f"Unique politicians: {df['Name'].nunique()}")
        logger.info(f"Unique tickers: {df['Ticker'].nunique()}")
        logger.info(f"Chamber breakdown: {df['Chamber'].value_counts().to_dict()}")
        logger.info(f"Party breakdown: {df['Party'].value_counts().to_dict()}")
        
        self.clean_trades = df
        return df
    
    def _parse_trade_amounts(self, df):
        """
        Parse transaction amounts from string ranges
        """
        
        def parse_amount_range(amount_str):
            """Parse a single amount string"""
            if pd.isna(amount_str):
                return None, None, None
            
            try:
                # Remove $, commas, spaces
                clean = str(amount_str).replace('$', '').replace(',', '').strip()
                
                # Handle different formats
                if '-' in clean:
                    parts = clean.split('-')
                elif ' to ' in clean.lower():
                    parts = clean.lower().split('to')
                elif 'over' in clean.lower():
                    parts = [clean.split()[-1], str(float(clean.split()[-1]) * 2)]
                else:
                    try:
                        val = float(clean)
                        return val, val, min(val, 250000)
                    except:
                        return None, None, None
                
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                midpoint = (min_val + max_val) / 2
                trade_weight = min(midpoint, 250000)
                
                return min_val, max_val, trade_weight
                
            except Exception as e:
                logger.debug(f"Could not parse amount: {amount_str} - {e}")
                return None, None, None
        
        # Apply parsing row by row
        amounts_data = []
        for amount_str in df['Trade_Size_USD']:
            amounts_data.append(parse_amount_range(amount_str))
        
        amounts_df = pd.DataFrame(amounts_data, columns=['amount_min', 'amount_max', 'trade_weight'])
        
        df = df.reset_index(drop=True)
        df['amount_min'] = amounts_df['amount_min']
        df['amount_max'] = amounts_df['amount_max']
        df['trade_weight'] = amounts_df['trade_weight']
        
        before = len(df)
        df = df[df['trade_weight'].notna()]
        logger.info(f"After amount parsing: {len(df)} (removed {before-len(df)} with invalid amounts)")
        
        return df
    
    def load_committee_assignments(self):
        """
        Load committee assignments from unitedstates/congress-legislators
        
        Handles:
        - Main committees (e.g., SSAF)
        - Subcommittees (e.g., SSAF13) - rolled up to parent
        - Multiple committee memberships per person
        """
        
        local_file = '/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw/committee-membership-current.yaml'
        
        try:
            with open(local_file, 'r') as f:
                committee_data = yaml.safe_load(f)
            logger.info("Loaded committee data from local YAML file")
        except FileNotFoundError:
            logger.info("Local file not found, fetching from GitHub...")
            url = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/committee-membership-current.yaml"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                committee_data = yaml.safe_load(response.text)
                
                os.makedirs('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw', exist_ok=True)
                with open(local_file, 'w') as f:
                    yaml.dump(committee_data, f)
                logger.info("Saved committee data to local YAML file")
            except Exception as e:
                logger.error(f"Could not fetch committee data: {e}")
                logger.info("Continuing without committee data...")
                return pd.DataFrame()
        
        # Committee name mappings
        committee_names = {
            # Senate Committees
            'SSAF': 'Senate Committee on Agriculture, Nutrition, and Forestry',
            'SSAP': 'Senate Committee on Appropriations',
            'SSAS': 'Senate Committee on Armed Services',
            'SSBK': 'Senate Committee on Banking, Housing, and Urban Affairs',
            'SSBU': 'Senate Committee on Budget',
            'SSCM': 'Senate Committee on Commerce, Science, and Transportation',
            'SSEG': 'Senate Committee on Energy and Natural Resources',
            'SSEV': 'Senate Committee on Environment and Public Works',
            'SSFI': 'Senate Committee on Finance',
            'SSFR': 'Senate Committee on Foreign Relations',
            'SSGA': 'Senate Committee on Homeland Security and Governmental Affairs',
            'SSHR': 'Senate Committee on Health, Education, Labor, and Pensions',
            'SSJU': 'Senate Committee on the Judiciary',
            'SSRA': 'Senate Committee on Rules and Administration',
            'SSSB': 'Senate Committee on Small Business and Entrepreneurship',
            'SSVT': 'Senate Committee on Veterans Affairs',
            'SLIN': 'Senate Select Committee on Intelligence',
            'SSEC': 'Senate Select Committee on Ethics',
            'SSAT': 'Senate Special Committee on Aging',
            'SSIE': 'Senate Committee on Indian Affairs',
            # House Committees
            'HSAG': 'House Committee on Agriculture',
            'HSAP': 'House Committee on Appropriations',
            'HSAS': 'House Committee on Armed Services',
            'HSBU': 'House Committee on Budget',
            'HSED': 'House Committee on Education and the Workforce',
            'HSIF': 'House Committee on Energy and Commerce',
            'HSBA': 'House Committee on Financial Services',
            'HSFA': 'House Committee on Foreign Affairs',
            'HSGO': 'House Committee on Oversight and Accountability',
            'HSHM': 'House Committee on Homeland Security',
            'HSHA': 'House Committee on House Administration',
            'HSJU': 'House Committee on the Judiciary',
            'HSII': 'House Committee on Natural Resources',
            'HSRU': 'House Committee on Rules',
            'HSSY': 'House Committee on Science, Space, and Technology',
            'HSSM': 'House Committee on Small Business',
            'HSPW': 'House Committee on Transportation and Infrastructure',
            'HSVR': 'House Committee on Veterans Affairs',
            'HSWM': 'House Committee on Ways and Means',
            'HSIG': 'House Permanent Select Committee on Intelligence',
            'HLIG': 'House Permanent Select Committee on Intelligence',
            # Joint Committees
            'JSEC': 'Joint Economic Committee',
            'JSLC': 'Joint Committee on the Library',
            'JSPR': 'Joint Committee on Printing',
            'JSTX': 'Joint Committee on Taxation',
        }
        
        def extract_parent_committee(committee_code):
            """Extract parent committee from code (SSAF13 -> SSAF)"""
            parent = re.match(r'^([A-Z]+)', committee_code)
            if parent:
                return parent.group(1)
            return committee_code
        
        # Parse committee assignments
        assignments = []
        parent_committees_found = set()
        subcommittees_found = set()
        
        logger.info(f"Total committee/subcommittee entries: {len(committee_data)}")
        
        if isinstance(committee_data, dict):
            for committee_code, members_list in committee_data.items():
                parent_code = extract_parent_committee(committee_code)
                
                if committee_code == parent_code:
                    parent_committees_found.add(parent_code)
                else:
                    subcommittees_found.add(committee_code)
                
                committee_name = committee_names.get(parent_code, parent_code)
                
                if parent_code.startswith('SS'):
                    committee_type = 'senate'
                elif parent_code.startswith('HS') or parent_code.startswith('HL'):
                    committee_type = 'house'
                elif parent_code.startswith('JS'):
                    committee_type = 'joint'
                else:
                    committee_type = 'unknown'
                
                if not isinstance(members_list, list):
                    continue
                
                for member in members_list:
                    if not isinstance(member, dict):
                        continue
                    
                    bioguide_id = member.get('bioguide', '')
                    if not bioguide_id:
                        continue
                    
                    title = str(member.get('title', '')).lower()
                    is_subcommittee = (committee_code != parent_code)
                    
                    if 'chairman' in title or ('chair' in title and 'ranking' not in title and 'vice' not in title):
                        leadership = 'Subcommittee Chair' if is_subcommittee else 'Chair'
                    elif 'ranking member' in title or 'ranking' in title:
                        leadership = 'Subcommittee Ranking Member' if is_subcommittee else 'Ranking Member'
                    elif 'vice' in title and 'chair' in title:
                        leadership = 'Vice Chair'
                    else:
                        leadership = 'Member'
                    
                    party_raw = member.get('party', 'unknown')
                    
                    assignments.append({
                        'bioguide_id': bioguide_id,
                        'committee_code': parent_code,
                        'original_code': committee_code,
                        'committee_name': committee_name,
                        'committee_type': committee_type,
                        'is_subcommittee': is_subcommittee,
                        'leadership_role': leadership,
                        'party': party_raw,
                        'title': member.get('title', ''),
                        'name': member.get('name', ''),
                        'rank': member.get('rank', 999)
                    })
        
        if not assignments:
            logger.warning("No committee assignments parsed!")
            return pd.DataFrame()
        
        df_committees = pd.DataFrame(assignments)
        
        # Consolidate: keep highest leadership role per person-committee
        def get_leadership_priority(role):
            priority = {
                'Chair': 5,
                'Ranking Member': 4,
                'Vice Chair': 3,
                'Subcommittee Chair': 2,
                'Subcommittee Ranking Member': 1,
                'Member': 0
            }
            return priority.get(role, 0)
        
        df_committees['leadership_priority'] = df_committees['leadership_role'].apply(get_leadership_priority)
        df_committees = df_committees.sort_values('leadership_priority', ascending=False)
        df_committees = df_committees.drop_duplicates(subset=['bioguide_id', 'committee_code'], keep='first')
        df_committees = df_committees.drop('leadership_priority', axis=1)
        
        logger.info(f"Total assignments (after deduplication): {len(df_committees)}")
        logger.info(f"Unique members: {df_committees['bioguide_id'].nunique()}")
        logger.info(f"Unique parent committees: {df_committees['committee_code'].nunique()}")
        logger.info(f"Parent committees found: {len(parent_committees_found)}")
        logger.info(f"Subcommittees rolled up: {len(subcommittees_found)}")
        
        logger.info(f"\nTop 15 committees by membership:")
        logger.info(df_committees['committee_name'].value_counts().head(15))
        
        return df_committees
    
    def merge_trades_with_committees(self, trades_df, committees_df):
        """
        Merge trade data with committee assignments
        Handles multiple committees per politician
        """
        
        if len(committees_df) == 0:
            logger.warning("No committee data available - adding placeholder columns")
            trades_df['committees_list'] = [['Unknown']] * len(trades_df)
            trades_df['leadership_roles'] = [['Member']] * len(trades_df)
            trades_df['committee_types'] = [['unknown']] * len(trades_df)
            trades_df['committee_codes'] = [['Unknown']] * len(trades_df)
            trades_df['primary_committee'] = 'Unknown'
            trades_df['primary_leadership'] = 'Member'
            trades_df['num_committees'] = 0
            return trades_df
        
        # Aggregate all committees per politician
        committee_agg = committees_df.groupby('bioguide_id').agg({
            'committee_name': lambda x: list(x),
            'committee_code': lambda x: list(x),
            'leadership_role': lambda x: list(x),
            'committee_type': lambda x: list(x),
            'party': 'first'
        }).reset_index()
        
        committee_agg.columns = ['bioguide_id', 'committees_list', 'committee_codes',
                                 'leadership_roles', 'committee_types', 'committee_party']
        
        merged = trades_df.merge(committee_agg, left_on='BioGuideID', right_on='bioguide_id', how='left')
        
        # Fill missing data
        merged['committees_list'] = merged['committees_list'].apply(
            lambda x: x if isinstance(x, list) else ['Unknown']
        )
        merged['committee_codes'] = merged['committee_codes'].apply(
            lambda x: x if isinstance(x, list) else ['Unknown']
        )
        merged['leadership_roles'] = merged['leadership_roles'].apply(
            lambda x: x if isinstance(x, list) else ['Member']
        )
        merged['committee_types'] = merged['committee_types'].apply(
            lambda x: x if isinstance(x, list) else ['unknown']
        )
        
        # Primary committee (first one for now)
        merged['primary_committee'] = merged['committees_list'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
        )
        merged['primary_leadership'] = merged['leadership_roles'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Member'
        )
        merged['num_committees'] = merged['committees_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        trades_with_committees = (merged['primary_committee'] != 'Unknown').sum()
        pct_with_committees = trades_with_committees / len(merged) * 100
        
        logger.info(f"\nMerge statistics:")
        logger.info(f"Total trades: {len(merged)}")
        logger.info(f"Trades with committee data: {trades_with_committees} ({pct_with_committees:.1f}%)")
        logger.info(f"Average committees per politician: {merged['num_committees'].mean():.1f}")
        
        return merged
    
    def calculate_seniority(self, df):
        """
        Calculate politician seniority (years in Congress)
        Approximation based on first trade date
        """
        
        first_appearance = df.groupby('Name')['Traded'].min().reset_index()
        first_appearance.columns = ['Name', 'first_trade_date']
        
        df = df.merge(first_appearance, on='Name', how='left')
        df['seniority_years'] = (df['Traded'] - df['first_trade_date']).dt.days / 365.25
        df['seniority_years'] = df['seniority_years'] + 5  # Add base years
        
        logger.info(f"Seniority statistics:")
        logger.info(df['seniority_years'].describe())
        
        return df
    
    def prepare_for_csv_export(self, df):
        """
        Convert list columns to pipe-delimited strings for clean CSV export
        
        Converts:
        ['Committee A', 'Committee B'] -> "Committee A|Committee B"
        """
        
        df = df.copy()
        
        list_columns = ['committees_list', 'committee_codes', 'leadership_roles', 'committee_types']
        
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: '|'.join(x) if isinstance(x, list) else str(x)
                )
        
        logger.info("List columns converted to pipe-delimited strings")
        
        return df


# Main execution
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed', exist_ok=True)
    os.makedirs('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw', exist_ok=True)
    
    loader = CongressionalDataLoader(trades_file='/Users/elikeldsen/Documents/Research/policy-informed-trading/data/raw/congresstradingall.csv')
    
    # Step 1: Load raw data
    print("STEP 1: Raw data loaded")
    print("-"*10)
    raw_trades = loader.load_raw_trades()
    
    # Step 2: Apply screening
    print("STEP 2: Screening criteria apply")
    print("-"*10)
    clean_trades = loader.apply_ziobrowski_screening()
    
    if len(clean_trades) == 0:
        print("No trades passed screening")
        print("-"*10)
        import sys
        sys.exit(1)
    
    # Step 3: Load committees
    print("STEP 3: Committee assignments load")
    print("-"*10)
    committees = loader.load_committee_assignments()
    
    # Step 4: Merge
    print("STEP 4: Merging trades with committees")
    print("-"*10)
    enhanced_trades = loader.merge_trades_with_committees(clean_trades, committees)
    
    # Step 5: Calculate seniority
    print("STEP 5: Seniority calculations")
    print("-"*10)
    final_trades = loader.calculate_seniority(enhanced_trades)
    
    # Step 6: Filter for quality
    print("STEP 6: Filtering data for qualiity")
    print("-"*10)
    
    before_filter = len(final_trades)
    trades_with_unknown = (final_trades['primary_committee'] == 'Unknown').sum()
    
    logger.info(f"Before committee filter: {before_filter:,} trades")
    logger.info(f"Trades with unknown committees: {trades_with_unknown:,} ({trades_with_unknown/before_filter*100:.1f}%)")
    
    # Filter for quality
    quality_trades = final_trades[
        (final_trades['primary_committee'] != 'Unknown') &
        (final_trades['primary_committee'].notna()) &
        (final_trades['primary_committee'] != '') &
        (final_trades['num_committees'] > 0)
    ].copy()
    
    after_filter = len(quality_trades)
    pct_kept = (after_filter / before_filter) * 100
    
    logger.info(f"\nAfter committee filter: {after_filter:,} trades")
    logger.info(f"Kept: {pct_kept:.1f}% of trades")
    logger.info(f"Removed: {before_filter - after_filter:,} trades without committee data")
    
    logger.info(f"\nQuality dataset statistics:")
    logger.info(f"  Unique politicians: {quality_trades['Name'].nunique()}")
    logger.info(f"  Average committees per politician: {quality_trades['num_committees'].mean():.2f}")
    logger.info(f"\nTop committees by trade count:")
    logger.info(quality_trades['primary_committee'].value_counts().head(15))
    
    # Step 7: Save in multiple formats
    print("STEP 7: Saving filtered/organized data")
    print("-"*10)
    
    # Prepare CSV-friendly versions
    logger.info("Preparing CSV exports...")
    final_trades_csv = loader.prepare_for_csv_export(final_trades)
    quality_trades_csv = loader.prepare_for_csv_export(quality_trades)
    
    # Save FULL dataset
    logger.info("Saving full dataset (3 formats)...")
    final_trades_csv.to_csv('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_full.csv', index=False)
    final_trades.to_pickle('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_full.pkl')
    final_trades.to_parquet('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_full.parquet')
    
    # Save QUALITY dataset
    logger.info("Saving quality dataset (3 formats)...")
    quality_trades_csv.to_csv('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.csv', index=False)
    quality_trades.to_pickle('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.pkl')
    quality_trades.to_parquet('/Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.parquet')
    
    print(f"\n FULL DATASET (all {len(final_trades):,} trades):")
    print(f"   CSV:     /Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_full.csv")
    print(f"   Pickle:  /Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_full.pkl")
    print(f"   Parquet: /Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_full.parquet")
    
    print(f"\n QUALITY DATASET ({len(quality_trades):,} trades with committee data):")
    print(f"   CSV:     /Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.csv")
    print(f"   Pickle:  /Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.pkl")
    print(f"   Parquet: /Users/elikeldsen/Documents/Research/policy-informed-trading/data/processed/trades_with_committees_quality.parquet")
    
    print(f"\n QUALITY DATASET SUMMARY:")
    print(f"   Date range: {quality_trades['Traded'].min().date()} to {quality_trades['Traded'].max().date()}")
    print(f"   Politicians: {quality_trades['Name'].nunique():,}")
    print(f"   Tickers: {quality_trades['Ticker'].nunique():,}")
    print(f"   Senate: {(quality_trades['Chamber'] == 'Senate').sum():,} trades")
    print(f"   House: {(quality_trades['Chamber'] == 'House').sum():,} trades")
    print(f"   Buys: {(quality_trades['direction'] == 1).sum():,}")
    print(f"   Sells: {(quality_trades['direction'] == -1).sum():,}")
    print(f"   Avg committees/politician: {quality_trades['num_committees'].mean():.2f}")