#!/usr/bin/env python3
"""
Multi-Signal Weak Supervision Framework for Trade Informativeness.
Addresses disclosure delay challenge using structural signals.
Inspired by weak supervision methodology (Ratner et al., Stanford/NeurIPS).

Author: ShaneStreet
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeakSupervisionLabeler:
    """
    Multi-signal weak supervision for identifying informed trades
    
    Key Innovation: Combines multiple noisy signals to generate high-quality
    labels WITHOUT waiting for subsequent returns. This addresses the 30-45 day
    disclosure delay by using immediate structural features.
    
    Signals:
    1. Committee relevance (immediate)
    2. Leadership position (immediate)
    3. Seniority + chamber (immediate, Ziobrowski finding)
    4. Trade timing relative to events (immediate)
    5. Trade size percentile (immediate)
    6. Subsequent returns (delayed but useful for training)
    """
    
    def __init__(self, 
                 relevance_high_threshold: float = 0.75,
                 relevance_low_threshold: float = 0.30,
                 seniority_threshold: float = 16.0,
                 use_return_signal: bool = True):
        """
        Initialize weak supervision labeler
        Args:
            relevance_high_threshold: Committee relevance score indicating informed trade
            relevance_low_threshold: Committee relevance score indicating uninformed trade
            seniority_threshold: Years of seniority for senior politician signal
            use_return_signal: Whether to include subsequent returns as a signal
        """
        self.relevance_high = relevance_high_threshold
        self.relevance_low = relevance_low_threshold
        self.seniority_threshold = seniority_threshold
        self.use_return_signal = use_return_signal
        
        self.signal_stats = {}
    
    def signal_1_high_relevance(self, row) -> Optional[int]:
        """
        Signal 1: High committee-sector relevance
        
        Rationale: Trades in sectors directly under politician's committee
        jurisdiction are more likely to be informed (domain expertise)
        """
        if not hasattr(row, 'relevance_score'):
            return None
            
        if row.relevance_score >= self.relevance_high:
            return 1  # INFORMED
        elif row.relevance_score < self.relevance_low:
            return 0  # UNINFORMED
        return None  # ABSTAIN
    
    def signal_2_leadership_position(self, row) -> Optional[int]:
        """
        Signal 2: Committee leadership + relevance
        
        Rationale: Committee chairs and ranking members have superior
        information access (Ziobrowski et al. 2004)
        """
        if not hasattr(row, 'primary_leadership') or not hasattr(row, 'relevance_score'):
            return None
        
        leadership_roles = ['Chair', 'Ranking Member', 'Vice Chair', 
                           'Subcommittee Chair', 'Subcommittee Ranking Member']
        
        if row.primary_leadership in leadership_roles:
            # Leaders need moderate relevance to be informed
            if row.relevance_score >= 0.60:
                return 1  # informed
            elif row.relevance_score < 0.30:
                return 0  # uninformed
        
        return None  # abstain
    
    def signal_3_senate_seniority(self, row) -> Optional[int]:
        """
        Signal 3: Senate elite (Ziobrowski's key finding)
        
        Rationale: Senior senators (>16 years) with high relevance
        significantly outperform (Ziobrowski et al. 2004, Table 5)
        """
        if not hasattr(row, 'Chamber') or not hasattr(row, 'seniority_years'):
            return None
        
        if not hasattr(row, 'relevance_score'):
            return None
        
        if row.Chamber == 'Senate' and row.seniority_years >= self.seniority_threshold:
            if row.relevance_score >= 0.70:
                return 1  # informed
        
        # Junior politicians or low relevance
        if hasattr(row, 'seniority_years') and row.seniority_years < 7:
            if row.relevance_score < 0.40:
                return 0  # uninformed
        
        return None  # abstain
    
    def signal_4_trade_timing(self, row) -> Optional[int]:
        """
        Signal 4: Proximity to information events
        
        Rationale: Trades close to earnings announcements or legislative
        action suggest information-driven timing
        """
        has_earnings = hasattr(row, 'days_to_earnings')
        has_legislative = hasattr(row, 'days_to_legislative_action')
        
        if not has_earnings and not has_legislative:
            return None
        
        # Check earnings proximity
        if has_earnings and not pd.isna(row.days_to_earnings):
            if row.days_to_earnings <= 30:
                return 1  # informed (close to earnings)
            elif row.days_to_earnings > 90:
                return 0  # uninformed (far from earnings)
        
        # Check legislative proximity
        if has_legislative and not pd.isna(row.days_to_legislative_action):
            if row.days_to_legislative_action <= 30:
                return 1  # informed (close to legislative action)
        
        return None  # abstain
    
    def signal_5_large_conviction_trade(self, row) -> Optional[int]:
        """
        Signal 5: Large trade size + high relevance = strong conviction
        
        Rationale: Politicians putting significant capital in their area of
        expertise suggests high confidence in information
        """
        if not hasattr(row, 'trade_size_percentile') or not hasattr(row, 'relevance_score'):
            return None
        
        if pd.isna(row.trade_size_percentile):
            return None
        
        # Large trade (top 20%) + high relevance = informed
        if row.trade_size_percentile >= 0.80 and row.relevance_score >= 0.70:
            return 1  # informed
        
        # Small trade + low relevance = less informed
        if row.trade_size_percentile <= 0.20 and row.relevance_score < 0.40:
            return 0  # uninformed
        
        return None  # abstain
    
    def signal_6_subsequent_performance(self, row) -> Optional[int]:
        """
        Signal 6: Subsequent returns (noisy potentially useful)
        
        NOTE: This signal has DELAY - only use for historical training data
        """
        if not self.use_return_signal:
            return None
        
        # Check for any return column
        return_col = None
        for col in ['subsequent_6m_return', 'subsequent_return', 'holding_period_return']:
            if hasattr(row, col):
                return_col = col
                break
        
        if return_col is None:
            return None
        
        ret = getattr(row, return_col)
        if pd.isna(ret):
            return None
        
        # Conservative thresholds (signal is noisy)
        if ret > 0.20:
            return 1  # informed
        elif ret < -0.15:
            return 0  # uninformed
        
        return None  # abstain
    
    def generate_label_for_trade(self, row) -> Dict:
        """
        Generate weak supervision label for a single trade
        
        Returns:
            dict with:
                - 'label': 1 (informed), 0 (uninformed), or None (abstain)
                - 'confidence': 0-1 score based on vote agreement
                - 'votes': list of individual signal votes
                - 'num_signals': how many signals voted
        """
        # Collect votes from all signals
        votes = []
        
        signal_1 = self.signal_1_high_relevance(row)
        if signal_1 is not None:
            votes.append(signal_1)
        
        signal_2 = self.signal_2_leadership_position(row)
        if signal_2 is not None:
            votes.append(signal_2)
        
        signal_3 = self.signal_3_senate_seniority(row)
        if signal_3 is not None:
            votes.append(signal_3)
        
        signal_4 = self.signal_4_trade_timing(row)
        if signal_4 is not None:
            votes.append(signal_4)
        
        signal_5 = self.signal_5_large_conviction_trade(row)
        if signal_5 is not None:
            votes.append(signal_5)
        
        signal_6 = self.signal_6_subsequent_performance(row)
        if signal_6 is not None:
            votes.append(signal_6)
        
        # Aggregate votes
        if len(votes) == 0:
            return {
                'label': None,
                'confidence': 0.0,
                'votes': [],
                'num_signals': 0
            }
        
        # Majority vote
        vote_sum = sum(votes)
        vote_avg = vote_sum / len(votes)
        
        # Determine label
        if vote_avg > 0.5:
            label = 1  # informed
        elif vote_avg < 0.5:
            label = 0  # uninfored
        else:
            label = None  # TIE - abstain
        
        # Confidence based on vote agreement
        # 100% agreement = 1.0, 50% agreement = 0.0
        confidence = abs(vote_avg - 0.5) * 2.0
        
        return {
            'label': label,
            'confidence': confidence,
            'votes': votes,
            'num_signals': len(votes)
        }
    
    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weak supervision to entire dataset
        
        Args:
            df: DataFrame with trade data
            
        Returns:
            DataFrame with added columns:
                - ws_label: weak supervision label (0/1/None)
                - ws_confidence: confidence score (0-1)
                - ws_num_signals: number of signals that voted
        """
        
        results = []
        for idx, row in df.iterrows():
            result = self.generate_label_for_trade(row)
            results.append(result)
        
        # Add to dataframe
        df = df.copy()
        df['ws_label'] = [r['label'] for r in results]
        df['ws_confidence'] = [r['confidence'] for r in results]
        df['ws_num_signals'] = [r['num_signals'] for r in results]
        
        # Calculate statistics
        self._calculate_statistics(df, results)
        
        return df
    
    def _calculate_statistics(self, df: pd.DataFrame, results: List[Dict]):
        """Calculate and log statistics about labeling"""
        
        total = len(df)
        labeled = df['ws_label'].notna().sum()
        informed = (df['ws_label'] == 1).sum()
        uninformed = (df['ws_label'] == 0).sum()
        abstained = total - labeled
        
        logger.info(f"\n{'='*80}")
        logger.info(f"WEAK SUPERVISION STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"\nCoverage:")
        logger.info(f"  Total trades: {total:,}")
        logger.info(f"  Labeled: {labeled:,} ({labeled/total*100:.1f}%)")
        logger.info(f"  Abstained: {abstained:,} ({abstained/total*100:.1f}%)")
        
        logger.info(f"\nLabel Distribution:")
        logger.info(f"  Informed (1): {informed:,} ({informed/labeled*100:.1f}% of labeled)")
        logger.info(f"  Uninformed (0): {uninformed:,} ({uninformed/labeled*100:.1f}% of labeled)")
        
        logger.info(f"\nConfidence:")
        labeled_df = df[df['ws_label'].notna()]
        logger.info(f"  Mean confidence: {labeled_df['ws_confidence'].mean():.3f}")
        logger.info(f"  Median confidence: {labeled_df['ws_confidence'].median():.3f}")
        
        logger.info(f"\nSignal Coverage:")
        logger.info(f"  Mean signals per trade: {df['ws_num_signals'].mean():.2f}")
        logger.info(f"  Median signals per trade: {df['ws_num_signals'].median():.0f}")
        
        # High confidence labels
        high_conf = (labeled_df['ws_confidence'] >= 0.8).sum()
        logger.info(f"\nHigh Confidence Labels (≥0.8):")
        logger.info(f"  Count: {high_conf:,} ({high_conf/labeled*100:.1f}% of labeled)")
        
        logger.info(f"\n{'='*80}")
    
    def create_comparison_report(self, df: pd.DataFrame, 
                                 return_column: str = 'subsequent_6m_return') -> pd.DataFrame:
        """
        Create comparison report: WS labels vs. actual returns
        
        This validates that weak supervision labels correlate with performance
        """
        if return_column not in df.columns:
            logger.warning(f"Return column '{return_column}' not found. Skipping comparison.")
            return None
        
        labeled_df = df[df['ws_label'].notna()].copy()
        labeled_df = labeled_df[labeled_df[return_column].notna()]
        
        if len(labeled_df) == 0:
            logger.warning("No trades with both WS labels and returns. Skipping comparison.")
            return None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"WEAK SUPERVISION VALIDATION")
        logger.info(f"{'='*80}")
        
        # informed vs uninformed returns
        informed_returns = labeled_df[labeled_df['ws_label'] == 1][return_column]
        uninformed_returns = labeled_df[labeled_df['ws_label'] == 0][return_column]
        
        # t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(informed_returns, uninformed_returns)
        logger.info(f"\n Statistical Test:")
        logger.info(f"t-statistic: {t_stat:.3f}")
        logger.info(f"p-value: {p_value:.4f}")
        if p_value < 0.05:
            logger.info(f"Significant at 5% level")
        else:
            logger.info(f"Not significant")
        
        return labeled_df


def main():
    """
    Application of weak supervision framework
    """
    print("\n" + "="*80)
    print("Weak Supervision Implementation")
    print("="*80)
    
    # Create sample trade
    sample_trade = pd.Series({
        'Ticker': 'NVDA',
        'relevance_score': 0.85,
        'primary_leadership': 'Chair',
        'Chamber': 'Senate',
        'seniority_years': 18,
        'days_to_earnings': 25,
        'trade_size_percentile': 0.90,
        'subsequent_6m_return': 0.28
    })
    
    labeler = WeakSupervisionLabeler()
    result = labeler.generate_label_for_trade(sample_trade)
    
    print("\nSample Trade:")
    print(f"  Ticker: {sample_trade['Ticker']}")
    print(f"  Committee Relevance: {sample_trade['relevance_score']:.2f}")
    print(f"  Leadership: {sample_trade['primary_leadership']}")
    print(f"  Chamber: {sample_trade['Chamber']}")
    print(f"  Seniority: {sample_trade['seniority_years']} years")
    
    print("\nWeak Supervision Result:")
    print(f"  Label: {result['label']} ({'INFORMED' if result['label']==1 else 'UNINFORMED'})")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Signals voted: {result['num_signals']}")
    print(f"  Votes: {result['votes']}")

if __name__ == "__main__":
    main()