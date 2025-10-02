#!/usr/bin/env python3
"""
Committee-Sector Relevance Mapper

Author: ShaneStreet
Date: 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommitteeSectorMapper:
    """
    Map congressional committees to stock sectors with relevance scores
    """
    
    def __init__(self):
        self.committee_sector_matrix = self._build_relevance_matrix()
        self.committee_weights = self._calculate_committee_weights()
        self.ticker_sector_cache = {}
        
        total_mappings = sum(len(v) for v in self.committee_sector_matrix.values())
        logger.info(f"  Committees mapped: {len(self.committee_sector_matrix)}")
        logger.info(f"  Sector relationships: {total_mappings}")
    
    def _build_relevance_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Build committee-sector relevance matrix with scores 0-1
        """
        
        matrix = {
            # SENATE BANKING, HOUSING, AND URBAN AFFAIRS
            'Senate Committee on Banking, Housing, and Urban Affairs': {
                'Financials': 0.95,
                'Banks': 0.95,
                'Diversified Banks': 0.95,
                'Regional Banks': 0.90,
                'Banks - Diversified': 0.95,
                'Banks - Regional': 0.90,
                'Insurance': 0.90,
                'Insurance - Life': 0.85,
                'Insurance - Property & Casualty': 0.90,
                'Insurance - Specialty': 0.85,
                'Insurance Brokers': 0.85,
                'Real Estate': 0.85,
                'Real Estate Services': 0.85,
                'Real Estate - Development': 0.85,
                'REIT': 0.80,
                'REIT - Diversified': 0.80,
                'REIT - Specialty': 0.80,
                'REIT - Residential': 0.80,
                'Financial Services': 0.90,
                'Financial Data & Stock Exchanges': 0.85,
                'Capital Markets': 0.85,
                'Credit Services': 0.85,
                'Asset Management': 0.80,
                'Investment Banking & Brokerage': 0.85,
                'Thrifts & Mortgage Finance': 0.90,
                'Investment Banking & Brokerage': 0.85,
            },
            
            # HOUSE FINANCIAL SERVICES
            'House Committee on Financial Services': {
                'Financials': 0.95,
                'Banks': 0.95,
                'Diversified Banks': 0.95,
                'Regional Banks': 0.90,
                'Banks - Diversified': 0.95,
                'Banks - Regional': 0.90,
                'Insurance': 0.90,
                'Insurance - Life': 0.85,
                'Insurance - Property & Casualty': 0.90,
                'Insurance Brokers': 0.85,
                'Real Estate': 0.85,
                'Real Estate Services': 0.85,
                'Financial Services': 0.90,
                'Financial Data & Stock Exchanges': 0.85,
                'Capital Markets': 0.85,
                'Credit Services': 0.85,
                'Asset Management': 0.80,
                'Investment Banking & Brokerage': 0.85,
            },
            
            # SENATE COMMERCE, SCIENCE, AND TRANSPORTATION
            'Senate Committee on Commerce, Science, and Transportation': {
                'Information Technology': 0.95,
                'Technology': 0.95,
                'Technology Hardware, Storage & Peripherals': 0.90,
                'Information Technology Services': 0.90,
                'Software': 0.90,
                'Software - Application': 0.90,
                'Software - Infrastructure': 0.90,
                'Application Software': 0.90,
                'Systems Software': 0.90,
                'Internet': 0.90,
                'Interactive Media & Services': 0.90,
                'Internet Content & Information': 0.90,
                'Internet & Direct Marketing Retail': 0.85,
                'Semiconductors': 0.85,
                'Semiconductors & Semiconductor Equipment': 0.85,
                'Semiconductor Equipment & Materials': 0.85,
                'Electronic Equipment': 0.80,
                'Electronic Components': 0.75,
                'Communications Equipment': 0.85,
                'Communication Equipment': 0.85,
                'Telecom Services': 0.85,
                'Telecommunications Services': 0.85,
                'IT Services': 0.85,
                'Airlines': 0.90,
                'Air Freight & Logistics': 0.85,
                'Integrated Freight & Logistics': 0.80,
                'Marine': 0.85,
                'Marine Shipping': 0.85,
                'Transportation': 0.85,
                'Railroads': 0.85,
                'Road & Rail': 0.80,
                'Auto Manufacturers': 0.75,
                'Auto Parts': 0.70,
                'Logistics': 0.75,
                'Travel Services': 0.70,
            },
            
            # HOUSE ENERGY AND COMMERCE
            'House Committee on Energy and Commerce': {
                'Information Technology': 0.95,
                'Technology': 0.95,
                'Technology Hardware, Storage & Peripherals': 0.90,
                'Information Technology Services': 0.90,
                'Software': 0.90,
                'Software - Application': 0.90,
                'Software - Infrastructure': 0.90,
                'Application Software': 0.90,
                'Systems Software': 0.90,
                'Internet': 0.90,
                'Interactive Media & Services': 0.90,
                'Internet & Direct Marketing Retail': 0.85,
                'Semiconductors': 0.85,
                'Semiconductors & Semiconductor Equipment': 0.85,
                'Telecom Services': 0.85,
                'Health Care': 0.90,
                'Health Care Equipment & Services': 0.90,
                'Health Care Providers & Services': 0.90,
                'Healthcare Plans': 0.85,
                'Health Care Plans': 0.85,
                'Medical Care Facilities': 0.85,
                'Pharmaceuticals': 0.90,
                'Drug Manufacturers': 0.90,
                'Drug Manufacturers - General': 0.90,
                'Drug Manufacturers - Specialty & Generic': 0.90,
                'Biotechnology': 0.85,
                'Medical Devices': 0.85,
                'Medical Instruments & Supplies': 0.85,
                'Health Care Equipment & Supplies': 0.85,
                'Diagnostics & Research': 0.80,
                'Medical Diagnostics & Research': 0.80,
                'Life Sciences Tools & Services': 0.80,
                'Managed Health Care': 0.85,
                'Energy': 0.85,
                'Oil & Gas': 0.85,
                'Oil, Gas & Consumable Fuels': 0.85,
                'Utilities': 0.80,
                'Utilities - Regulated Electric': 0.80,
            },
            
            # SENATE ENERGY AND NATURAL RESOURCES
            'Senate Committee on Energy and Natural Resources': {
                'Energy': 0.95,
                'Oil & Gas': 0.95,
                'Oil, Gas & Consumable Fuels': 0.95,
                'Oil & Gas E&P': 0.95,
                'Oil & Gas Exploration & Production': 0.95,
                'Oil & Gas Integrated': 0.95,
                'Oil & Gas Midstream': 0.90,
                'Oil & Gas Refining & Marketing': 0.90,
                'Oil & Gas Equipment & Services': 0.85,
                'Utilities': 0.90,
                'Utilities - Regulated Electric': 0.90,
                'Utilities - Regulated Gas': 0.90,
                'Utilities - Regulated Water': 0.85,
                'Utilities - Diversified': 0.85,
                'Utilities - Renewable': 0.85,
                'Utilities - Independent Power Producers': 0.85,
                'Coal': 0.85,
                'Coal & Consumable Fuels': 0.85,
                'Metals & Mining': 0.80,
                'Copper': 0.75,
                'Gold': 0.75,
                'Silver': 0.75,
                'Construction Materials': 0.70,
            },
            
            # SENATE ARMED SERVICES
            'Senate Committee on Armed Services': {
                'Aerospace & Defense': 0.95,
                'Defense': 0.95,
                'Industrials': 0.70,
                'Industrial Conglomerates': 0.70,
                'Conglomerates': 0.70,
                'Electrical Equipment': 0.65,
                'Electrical Equipment & Parts': 0.65,
                'Technology': 0.70,
                'Electronic Equipment & Instruments': 0.75,
                'Communication Equipment': 0.70,
            },
            
            # HOUSE ARMED SERVICES
            'House Committee on Armed Services': {
                'Aerospace & Defense': 0.95,
                'Defense': 0.95,
                'Industrials': 0.70,
                'Industrial Conglomerates': 0.70,
                'Conglomerates': 0.70,
            },
            
            # SENATE HEALTH, EDUCATION, LABOR, AND PENSIONS
            'Senate Committee on Health, Education, Labor, and Pensions': {
                'Health Care': 0.95,
                'Health Care Equipment & Services': 0.95,
                'Health Care Providers & Services': 0.95,
                'Healthcare Plans': 0.90,
                'Health Care Plans': 0.90,
                'Medical Care Facilities': 0.85,
                'Health Care Technology': 0.85,
                'Pharmaceuticals': 0.95,
                'Drug Manufacturers': 0.95,
                'Drug Manufacturers - General': 0.95,
                'Drug Manufacturers - Specialty & Generic': 0.95,
                'Biotechnology': 0.90,
                'Life Sciences Tools & Services': 0.80,
                'Medical Devices': 0.85,
                'Medical Instruments & Supplies': 0.85,
                'Health Care Equipment & Supplies': 0.85,
                'Medical Diagnostics & Research': 0.80,
                'Diagnostics & Research': 0.80,
                'Medical Distribution': 0.80,
                'Managed Health Care': 0.90,
            },
            
            # SENATE AGRICULTURE, NUTRITION, AND FORESTRY
            'Senate Committee on Agriculture, Nutrition, and Forestry': {
                'Food Products': 0.90,
                'Agricultural Products': 0.95,
                'Agricultural Inputs': 0.90,
                'Packaged Foods': 0.85,
                'Food & Staples Retailing': 0.75,
                'Consumer Staples': 0.70,
                'Farm Products': 0.95,
                'Farm & Heavy Construction Machinery': 0.80,
                'Beverages - Non-Alcoholic': 0.70,
                'Beverages - Alcoholic': 0.70,
            },
            
            # HOUSE AGRICULTURE
            'House Committee on Agriculture': {
                'Food Products': 0.90,
                'Agricultural Products': 0.95,
                'Agricultural Inputs': 0.90,
                'Packaged Foods': 0.85,
                'Food & Staples Retailing': 0.75,
                'Beverages - Non-Alcoholic': 0.70,
                'Farm Products': 0.95,
                'Restaurants': 0.65,
                'Farm & Heavy Construction Machinery': 0.80,
            },
            
            # SENATE JUDICIARY (Antitrust, IP)
            'Senate Committee on the Judiciary': {
                'Technology': 0.75,
                'Software': 0.75,
                'Software - Application': 0.75,
                'Software - Infrastructure': 0.75,
                'Internet': 0.80,
                'Interactive Media & Services': 0.80,
                'Internet & Direct Marketing Retail': 0.75,
                'Media': 0.75,
                'Entertainment': 0.75,
                'Broadcasting': 0.75,
                'Pharmaceuticals': 0.70,
                'Biotechnology': 0.65,
            },
            
            # HOUSE JUDICIARY
            'House Committee on the Judiciary': {
                'Technology': 0.75,
                'Software': 0.75,
                'Software - Application': 0.75,
                'Software - Infrastructure': 0.75,
                'Internet': 0.80,
                'Interactive Media & Services': 0.80,
                'Internet & Direct Marketing Retail': 0.75,
                'Media': 0.75,
                'Entertainment': 0.75,
            },
            
            # SENATE FINANCE (Taxes, Trade, Healthcare)
            'Senate Committee on Finance': {
                'Financials': 0.80,
                'Health Care': 0.80,
                'Pharmaceuticals': 0.75,
                'Drug Manufacturers - General': 0.75,
                'Healthcare Plans': 0.80,
                'Managed Health Care': 0.80,
                'Insurance': 0.85,
                'Insurance - Life': 0.85,
                'Insurance - Property & Casualty': 0.85,
                'Diversified Financial Services': 0.75,
            },
            
            # HOUSE WAYS AND MEANS (Taxes, Trade, Healthcare)
            'House Committee on Ways and Means': {
                'Financials': 0.80,
                'Health Care': 0.80,
                'Pharmaceuticals': 0.75,
                'Drug Manufacturers - General': 0.75,
                'Healthcare Plans': 0.80,
                'Managed Health Care': 0.80,
                'Insurance': 0.85,
                'Insurance - Life': 0.85,
                'Insurance - Property & Casualty': 0.85,
                'Insurance Brokers': 0.80,
            },
            
            # HOUSE TRANSPORTATION AND INFRASTRUCTURE
            'House Committee on Transportation and Infrastructure': {
                'Airlines': 0.90,
                'Air Freight & Logistics': 0.85,
                'Integrated Freight & Logistics': 0.85,
                'Marine': 0.85,
                'Marine Shipping': 0.85,
                'Transportation': 0.90,
                'Railroads': 0.90,
                'Road & Rail': 0.90,
                'Auto Manufacturers': 0.80,
                'Construction & Engineering': 0.85,
                'Engineering & Construction': 0.85,
                'Construction Materials': 0.75,
                'Building Products & Equipment': 0.75,
                'Industrial Conglomerates': 0.70,
                'Conglomerates': 0.70,
            },
            
            # SENATE/HOUSE INTELLIGENCE COMMITTEES
            'Senate Select Committee on Intelligence': {
                'Technology': 0.85,
                'Software': 0.80,
                'Software - Application': 0.80,
                'Software - Infrastructure': 0.80,
                'Systems Software': 0.80,
                'Information Technology Services': 0.80,
                'IT Services': 0.80,
                'Aerospace & Defense': 0.85,
                'Defense': 0.85,
                'Communications Equipment': 0.75,
                'Communication Equipment': 0.75,
                'Semiconductors': 0.75,
                'Cybersecurity': 0.85,
            },
            
            'House Permanent Select Committee on Intelligence': {
                'Technology': 0.85,
                'Software': 0.80,
                'Software - Application': 0.80,
                'Software - Infrastructure': 0.80,
                'Systems Software': 0.80,
                'Application Software': 0.80,
                'Information Technology Services': 0.80,
                'IT Services': 0.80,
                'Aerospace & Defense': 0.85,
                'Defense': 0.85,
                'Technology Hardware, Storage & Peripherals': 0.75,
                'Credit Services': 0.70,
            },
            
            # SENATE ENVIRONMENT AND PUBLIC WORKS (NEW)
            'Senate Committee on Environment and Public Works': {
                'Specialty Chemicals': 0.85,
                'Chemicals': 0.85,
                'Construction Materials': 0.75,
                'Building Products & Equipment': 0.75,
                'Utilities': 0.75,
                'Utilities - Regulated Water': 0.85,
                'Waste Management': 0.85,
                'Pollution & Treatment Controls': 0.85,
                'Engineering & Construction': 0.75,
            },
            
            # HOUSE OVERSIGHT AND ACCOUNTABILITY
            'House Committee on Oversight and Accountability': {
                'Internet & Direct Marketing Retail': 0.65,
                'Discount Stores': 0.60,
                'Specialty Retail': 0.60,
                'Restaurants': 0.60,
                'Household & Personal Products': 0.60,
                'Personal Products': 0.60,
                'Travel Services': 0.60,
                'Resorts & Casinos': 0.60,
                'Leisure': 0.60,
                'Auto Manufacturers': 0.60,
                'Consumer Cyclical': 0.55,
                'Consumer Defensive': 0.55,
            },
        }
        
        return matrix
    
    def _calculate_committee_weights(self) -> Dict[str, float]:
        """
        Weight committees by their information advantage potential
        """
        weights = {
            'Senate Committee on Banking, Housing, and Urban Affairs': 0.90,
            'House Committee on Financial Services': 0.90,
            'Senate Committee on Commerce, Science, and Transportation': 0.95,
            'House Committee on Energy and Commerce': 0.95,
            'Senate Committee on Energy and Natural Resources': 0.85,
            'Senate Committee on Armed Services': 0.90,
            'House Committee on Armed Services': 0.90,
            'Senate Committee on Health, Education, Labor, and Pensions': 0.85,
            'Senate Committee on Agriculture, Nutrition, and Forestry': 0.75,
            'House Committee on Agriculture': 0.75,
            'Senate Committee on the Judiciary': 0.80,
            'House Committee on the Judiciary': 0.80,
            'Senate Committee on Finance': 0.85,
            'House Committee on Ways and Means': 0.85,
            'House Committee on Transportation and Infrastructure': 0.80,
            'Senate Select Committee on Intelligence': 0.85,
            'House Permanent Select Committee on Intelligence': 0.85,
            'Senate Committee on Environment and Public Works': 0.75,
            'House Committee on Oversight and Accountability': 0.65,
        }
        return weights
    
    def get_stock_sector(self, ticker: str, use_cache: bool = True) -> str:
        """
        Get sector/industry for a stock
        """
        
        if use_cache and ticker in self.ticker_sector_cache:
            return self.ticker_sector_cache[ticker]
        
        # Manual mapping for common stocks
        common_stock_sectors = {
            # Technology
            'AAPL': 'Technology Hardware, Storage & Peripherals',
            'MSFT': 'Systems Software',
            'GOOGL': 'Interactive Media & Services', 'GOOG': 'Interactive Media & Services',
            'AMZN': 'Internet & Direct Marketing Retail',
            'META': 'Interactive Media & Services', 'FB': 'Interactive Media & Services',
            'NVDA': 'Semiconductors & Semiconductor Equipment',
            'TSM': 'Semiconductors & Semiconductor Equipment',
            'INTC': 'Semiconductors & Semiconductor Equipment',
            'AMD': 'Semiconductors & Semiconductor Equipment',
            'NFLX': 'Entertainment',
            'CSCO': 'Communications Equipment',
            'ORCL': 'Application Software',
            'ADBE': 'Application Software',
            'CRM': 'Application Software',
            'NOW': 'Application Software',
            
            # Financial
            'JPM': 'Diversified Banks', 'BAC': 'Diversified Banks',
            'WFC': 'Diversified Banks', 'C': 'Diversified Banks',
            'GS': 'Investment Banking & Brokerage',
            'MS': 'Investment Banking & Brokerage',
            'BLK': 'Asset Management',
            'USB': 'Banks - Regional', 'PNC': 'Banks - Regional',
            
            # Healthcare
            'JNJ': 'Pharmaceuticals', 'PFE': 'Pharmaceuticals',
            'UNH': 'Managed Health Care',
            'MRK': 'Pharmaceuticals', 'LLY': 'Pharmaceuticals',
            'ABBV': 'Pharmaceuticals',
            'ABT': 'Health Care Equipment & Supplies',
            'TMO': 'Life Sciences Tools & Services',
            'DHR': 'Health Care Equipment & Supplies',
            
            # Energy
            'XOM': 'Oil, Gas & Consumable Fuels',
            'CVX': 'Oil, Gas & Consumable Fuels',
            'COP': 'Oil, Gas & Consumable Fuels',
            'SLB': 'Oil & Gas Equipment & Services',
            'EOG': 'Oil & Gas E&P',
            
            # Defense
            'LMT': 'Aerospace & Defense', 'RTX': 'Aerospace & Defense',
            'BA': 'Aerospace & Defense', 'NOC': 'Aerospace & Defense',
            'GD': 'Aerospace & Defense', 'LHX': 'Aerospace & Defense',
        }
        
        if ticker in common_stock_sectors:
            sector = common_stock_sectors[ticker]
            self.ticker_sector_cache[ticker] = sector
            return sector
        
        # Try yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('industry', info.get('sector', 'Unknown'))
            
            if sector and sector != 'Unknown':
                self.ticker_sector_cache[ticker] = sector
                return sector
                
        except Exception as e:
            logger.debug(f"Could not fetch sector for {ticker}: {e}")
        
        self.ticker_sector_cache[ticker] = 'Unknown'
        return 'Unknown'
    
    def _fuzzy_sector_match(self, sector: str, sector_scores: Dict[str, float]) -> float:
        """
        Handles edge case mapping
        """
        
        if sector == 'Unknown':
            return 0.0
        
        # Check for exact match FIRST
        for matrix_sector, score in sector_scores.items():
            if sector.lower() == matrix_sector.lower():
                return score
        
        # Now do fuzzy matching
        sector_lower = sector.lower()
        sector_clean = sector_lower.replace(' - ', ' ').replace('&', '').replace(',', '')
        sector_words = set(sector_clean.split())
        
        best_score = 0.0
        
        for matrix_sector, score in sector_scores.items():
            matrix_lower = matrix_sector.lower()
            matrix_clean = matrix_lower.replace(' - ', ' ').replace('&', '').replace(',', '')
            matrix_words = set(matrix_clean.split())
            
            common_words = sector_words & matrix_words
            
            # 1 word match is enough
            if len(common_words) >= 1:
                overlap_ratio = len(common_words) / max(len(sector_words), len(matrix_words))
                
                # Less penalty for fuzzy matches
                fuzzy_score = score * (0.90 + 0.10 * overlap_ratio)  # Was 0.85
                
                if fuzzy_score > best_score:
                    best_score = fuzzy_score
        
        return best_score
    
    def calculate_relevance_score(self, 
                                  ticker: str,
                                  committees_list: List[str],
                                  leadership_roles: List[str],
                                  seniority_years: float,
                                  party: str = None) -> Tuple[float, str, Dict]:
        """
        Calculate relevance score for a trade
        """
        
        sector = self.get_stock_sector(ticker)
        
        max_relevance = 0.0
        best_committee = 'Unknown'
        best_leadership = 'Member'
        
        for i, committee in enumerate(committees_list):
            if committee == 'Unknown':
                continue
            
            if committee in self.committee_sector_matrix:
                sector_scores = self.committee_sector_matrix[committee]
                
                # Try exact match first
                committee_relevance = sector_scores.get(sector, 0.0)
                
                # If no exact match, try fuzzy matching
                if committee_relevance == 0.0:
                    committee_relevance = self._fuzzy_sector_match(sector, sector_scores)
                
                if committee_relevance > max_relevance:
                    max_relevance = committee_relevance
                    best_committee = committee
                    best_leadership = leadership_roles[i] if i < len(leadership_roles) else 'Member'
        
        if max_relevance == 0.0:
            max_relevance = 0.1
        
        committee_weight = self.committee_weights.get(best_committee, 0.5)
        
        leadership_multipliers = {
            'Chair': 1.8,
            'Ranking Member': 1.5,
            'Vice Chair': 1.3,
            'Subcommittee Chair': 1.2,
            'Subcommittee Ranking Member': 1.1,
            'Member': 1.0,
        }
        leadership_mult = leadership_multipliers.get(best_leadership, 1.0)
        
        if seniority_years < 7:
            seniority_mult = 1.3
        elif seniority_years < 15:
            seniority_mult = 1.0
        else:
            seniority_mult = 0.85
        
        relevance_score = (max_relevance * committee_weight * 
                          seniority_mult * leadership_mult)
        
        relevance_score = min(relevance_score, 1.0)
        
        explanation = {
            'ticker': ticker,
            'sector': sector,
            'best_committee': best_committee,
            'leadership': best_leadership,
            'base_relevance': max_relevance,
            'committee_weight': committee_weight,
            'seniority_mult': seniority_mult,
            'leadership_mult': leadership_mult,
            'final_score': relevance_score
        }
        
        return relevance_score, best_committee, explanation
    
    def calculate_relevance_for_dataframe(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relevance scores for entire DataFrame of trades
        """
        logger.info(f"Calculating relevance scores for {len(trades_df):,} trades...")
        
        required_cols = ['Ticker', 'committees_list', 'leadership_roles', 'seniority_years']
        missing_cols = [col for col in required_cols if col not in trades_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        results = []
        
        for idx, row in trades_df.iterrows():
            try:
                score, committee, explanation = self.calculate_relevance_score(
                    ticker=row['Ticker'],
                    committees_list=row['committees_list'],
                    leadership_roles=row['leadership_roles'],
                    seniority_years=row['seniority_years'],
                    party=row.get('Party', None)
                )
                
                results.append({
                    'relevance_score': score,
                    'most_relevant_committee': committee,
                    'sector': explanation['sector'],
                    'base_relevance': explanation['base_relevance'],
                    'committee_weight': explanation['committee_weight'],
                    'leadership_mult': explanation['leadership_mult'],
                    'seniority_mult': explanation['seniority_mult']
                })
                
            except Exception as e:
                logger.warning(f"Error calculating relevance for row {idx}: {e}")
                results.append({
                    'relevance_score': 0.1,
                    'most_relevant_committee': 'Unknown',
                    'sector': 'Unknown',
                    'base_relevance': 0.1,
                    'committee_weight': 0.5,
                    'leadership_mult': 1.0,
                    'seniority_mult': 1.0
                })
        
        results_df = pd.DataFrame(results)
        trades_with_scores = pd.concat([trades_df.reset_index(drop=True), results_df], axis=1)
        
        logger.info(f"  Mean relevance score: {results_df['relevance_score'].mean():.3f}")
        logger.info(f"  Median relevance score: {results_df['relevance_score'].median():.3f}")
        logger.info(f"  High relevance (>0.7): {(results_df['relevance_score'] > 0.7).sum():,} trades")
        logger.info(f"  Low relevance (<0.3): {(results_df['relevance_score'] < 0.3).sum():,} trades")
        
        return trades_with_scores


# Testing
if __name__ == "__main__":
    
    mapper = CommitteeSectorMapper()
    
    # Test fuzzy matching
    print("\nTest 1: 'Software - Application' should match matrix")
    mapper.ticker_sector_cache['TEST'] = 'Software - Application'
    score, committee, exp = mapper.calculate_relevance_score(
        ticker='TEST',
        committees_list=['House Committee on Energy and Commerce'],
        leadership_roles=['Member'],
        seniority_years=10
    )
    print(f"  Score: {score:.3f}, Committee: {committee}")
    print(f"  Sector matched: {exp['sector']}, Base relevance: {exp['base_relevance']:.3f}")
    
    # Test coverage of previously unknown sectors
    print("\nTest 2: Previously low-coverage sector - Telecom Services")
    mapper.ticker_sector_cache['TTEST'] = 'Telecom Services'
    score, committee, exp = mapper.calculate_relevance_score(
        ticker='TTEST',
        committees_list=['Senate Committee on Commerce, Science, and Transportation'],
        leadership_roles=['Chair'],
        seniority_years=12
    )
    print(f"  Score: {score:.3f}, Committee: {committee}")
    print(f"  Sector: {exp['sector']}, Base relevance: {exp['base_relevance']:.3f}")