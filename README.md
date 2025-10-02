# Policy-Informed Trading Strategy

**Systematic Framework for Identifying Information Advantage in Congressional Trading**

---

## Introduction

This strategy replicates and extends Ziobrowski et al. (2004), who discovered that U.S. Senators beat the market by 85 basis points per month. That's roughly 10% annually, which is impressive considering most professional fund managers struggle to beat the index.

The core innovation: **Committee-Sector Relevance Scoring**. We quantify each politician's informational advantage by matching their committee assignments to stock sectors. Senator on Banking Committee buying JPMorgan? High relevance (0.95). Same senator buying John Deere? Low relevance (0.15).

Unlike the original study which analyzed all congressional trades equally, we systematically filter for trades where politicians have the biggest edge—when they're trading in sectors they directly regulate. Think of it as insider trading's respectable cousin: leveraging publicly disclosed information about who knows what.

The methodology is academically rigorous (calendar-time portfolios, Fama-French 3-factor model, in-sample/out-of-sample validation) and fully reproducible. All code provided. No black boxes. No magic.

**Key Finding**: High relevance trades (relevance score ≥0.75) significantly outperform both low relevance trades and the S&P 500, with strong statistical significance and robust out-of-sample performance.

---

## Installation

### Prerequisites
```bash
pip install pandas numpy yfinance scipy matplotlib seaborn scikit-learn snorkel
```

### Required Data Files

1. **Congressional Trading Data**: `congresstradingall.csv`
   - Source: CapitolTrades, Quiver Quantitative, or similar
   - Must include: Name, Ticker, Transaction type, Trade date, Filed date, Transaction size

2. **Committee Assignments**: `committee-membership-current.yaml`
   - Source: github.com/unitedstates/congress-legislators
   - Auto-downloads if missing (script handles this)

3. **Price Data** (optional but recommended): `russell_3000_daily.parquet`
   - Russell 3000 daily price data for accurate returns
   - Without this, script downloads SPY from Yahoo Finance (slower, less comprehensive)
   - Must be obtained externally.

### Directory Structure
```
policy-informed-trading/
├── data/
│   ├── figures/                # Output figures here
│   ├── raw/                    # Place source files here
│   └── processed/              # Pipeline outputs go here
├── src/
│   ├── analysis/
│        ├── apply_weak_supervision.py
│        ├── calendar_time_portfolio_v4.py
│        ├── in_out_sample_comparison.py
│        └── weak_supervision_labels.py
│   ├── data/
│        ├── apply_relevance_scoring_v2.py
│        ├── committee_mapper_v2.py
│        └── data_loader.py
│   ├── results/
│        ├── results/           # Output figures here
│        ├── presentation_visualizations.py
│        └── visualizations.py
├── trading_env/
└── README.md
```

---

## Quick Start

Run the complete pipeline in sequence:

```bash
python data_loader.py                      # 1. Clean and merge committee data
python apply_relevance_scoring_v2.py       # 2. Calculate relevance scores
python apply_weak_supervision.py           # 3. Label informed trades
python calendar_time_portfolio_v4.py       # 4. Build portfolios, run regressions
python in_out_sample_comparison.py         # 5. Validate strategy robustness
python presentation_visualizations.py      # 6. Generate first round of charts
python visualizations.py                   # 7. Generate second round of charts
```

Each script saves intermediate outputs to `data/processed/`, so you can restart from any stage if needed.

---

## Workflow Overview

### Stage 1: Data Acquisition (data_loader.py)

Loads congressional trades and applies Ziobrowski's screening criteria:
- Common stocks only (no bonds, options, ETFs, or Senator Tuberville's latest cryptocurrency adventure)
- Valid tickers (surprisingly, proper spelling is optional on disclosure forms)
- Post-STOCK Act (April 4, 2012 onward—when disclosure became mandatory)
- Reasonable disclosure timing (0-180 days between trade and filing)
- Parseable transaction amounts (those "$15,001-$50,000" ranges are delightfully vague)

Merges with congressional committee assignments from the unitedstates/congress-legislators GitHub repository. Handles multiple committee memberships, leadership positions (Chair, Ranking Member), and calculates approximate seniority.

**Output**: Quality trades with full committee metadata.

### Stage 2: Committee-Sector Mapping (committee_mapper_v2.py)

Maps 19 major congressional committees to 200+ stock sectors with continuous relevance scores (0-1).

**Methodology**:

**Examples**:
- Senate Banking, Housing, Urban Affairs → Banks (0.95), Insurance (0.90), REITs (0.80)
- House Energy & Commerce → Semiconductors (0.85), Pharmaceuticals (0.90), Tech (0.95)
- Senate Armed Services → Aerospace & Defense (0.95)
- House Agriculture → Agricultural Products (0.95), Farm Machinery (0.80)

**Leadership Multipliers**: Chair (1.8x), Ranking Member (1.5x), Vice Chair (1.3x), Regular Member (1.0x)

**Seniority Adjustment**: Junior senators (<7 years) get 1.3x boost, seniors (>16 years) get 0.85x penalty. This follows Ziobrowski's empirical finding that junior senators outperform—they're hungrier and less distracted by re-election machinery.

Uses fuzzy string matching to handle sector naming inconsistencies between yfinance and the committee matrix.

### Stage 3: Relevance Scoring (apply_relevance_scoring_v2.py)

Calculates relevance score for each trade using the committee mapper. Assigns primary committee (highest relevance), primary leadership role, and stores all committee memberships for later analysis.

**Distribution**:
- High Relevance (≥0.75): 27% of trades
- Medium Relevance (0.3-0.75): 39% of trades
- Low Relevance (<0.3): 34% of trades

### Stage 4: Weak Supervision (weak_supervision_labels.py + apply_weak_supervision.py)

**Problem**: Can't wait 6 months for returns to identify informed trades.

**Solution**: Multi-signal weak supervision framework where six signals vote on each trade:

1. **High Committee Relevance** (≥0.75) → Vote: Informed
2. **Leadership Position** (Chair/Ranking + relevance ≥0.60) → Vote: Informed
3. **Senate Seniority Elite** (>16 years + relevance ≥0.70) → Vote: Informed
4. **Trade Timing** (within 30 days of earnings) → Vote: Informed
5. **Large Conviction Trade** (top 20% size + relevance ≥0.70) → Vote: Informed
6. **Subsequent Performance** (return >20%) → Vote: Informed (validation only)

Majority vote determines label. Confidence score measures vote unanimity (6/6 agree = 1.0 confidence, 4/6 agree = 0.67 confidence).

**Coverage**: 67.8% of trades labeled

This approach, inspired by Ratner et al.'s data programming work at Stanford, allows real-time trade classification without waiting for returns.

### Stage 5: Portfolio Construction (calendar_time_portfolio_v4.py)

Implements calendar-time portfolio methodology (Fama 1998, Mitchell & Stafford 2000):

1. Each day, portfolio includes all stocks traded in prior 126 days (6-month holding period)
2. Daily returns compound to monthly returns
3. Run Fama-French 3-factor regression:

```
R_portfolio - R_f = α + β₁(R_market - R_f) + β₂(SMB) + β₃(HML) + ε
```

Where α (alpha) is abnormal return, SMB is size factor, HML is value factor.

**Portfolios Constructed**:
- High Relevance (≥0.75)
- High Relevance (≥0.70) 
- Senate Elite (Senate + ≥0.75)
- Low Relevance (<0.3)
- Senate Buys vs House Buys
- All Buys (baseline)

**Metrics Calculated**: Alpha (monthly and annualized), t-statistics, p-values, Sharpe ratio, Sortino ratio, information ratio, maximum drawdown, win rate, Calmar ratio.

### Stage 6: Validation (in_out_sample_comparison.py)

Splits data at 2020-01-01 to test strategy robustness:
- **In-sample**: 2013-2019 (strategy development period)
- **Out-of-sample**: 2020-2025 (validation period—no peeking)

Tests three portfolios: High Relevance (≥0.75), Low Relevance (<0.3), S&P 500 benchmark.

**Good Signs**: Consistent alpha direction, Sharpe ratio within 20%, High Relevance beats Low Relevance in both periods.

**Red Flags**: Alpha reverses sign, Sharpe drops >50%, strategy only works in one period.

### Stage 7: Visualization (presentation_visualizations.py + create_workflow_diagram.py)

Generates figures:
- Performance metrics table (in-sample vs out-of-sample comparison)
- Cumulative returns charts (4 separate comparisons)
- Drawdown analysis
- Rolling 12-month alpha
- Return distributions
- Workflow diagram

---

## Key Results

High relevance portfolio (≥0.75) demonstrates:
- **Positive alpha**: Statistically significant in Fama-French 3-factor model
- **Sharpe ratio**: Beats both low relevance portfolio and S&P 500
- **Robust out-of-sample**: Performance persists in validation period (2020-2025)
- **Risk-adjusted outperformance**: Beats market after controlling for size, value, and beta

Low relevance portfolio (<0.3):
- Comparable to S&P 500
- Near-zero alpha
- Higher volatility, worse drawdowns
- Serves as control group—validates that indiscriminate copying of congressional trades doesn't work

---

## Limitations and Caveats

**Disclosure Lag**: Trades reported 30-45 days late. You're reading yesterday's news and hoping it still matters.

**Data Quality**: Handwritten disclosure forms, creative ticker spellings, broad transaction ranges. We handle most issues but perfection is impossible.

**STOCK Act Effect**: Post-2012 mandatory disclosure might have changed behavior. Politicians could be more careful now (or better at hiding things in blind trusts).

---

## References

**Primary Research**:
Ziobrowski, A.J., Cheng, P., Boyd, J.W., & Ziobrowski, B.J. (2004). "Abnormal Returns from the Common Stock Investments of the U.S. Senate." *Journal of Financial and Quantitative Analysis*, 39(4), 661-676.

**Methodology**:
- Fama, E.F. (1998). "Market Efficiency, Long-Term Returns, and Behavioral Finance." *Journal of Financial Economics*, 49, 283-306.
- Fama, E.F., & French, K.R. (1993). "Common Risk Factors in Returns on Stocks and Bonds." *Journal of Financial Economics*, 33, 3-56.
- Mitchell, M.L., & Stafford, E. (2000). "Managerial Decisions and Long-Term Stock Price Performance." *Journal of Business*, 73, 287-320.

**Data Sources**:
- Congressional trading disclosures via STOCK Act (2012)
- Committee assignments: github.com/unitedstates/congress-legislators
- Price data: Russell 3000 or Yahoo Finance (SPY)

---

## Disclaimer

This is for educational and research purposes only. Not investment advice. Congressional trading is legal for them, questionable for you. Past performance doesn't guarantee future results.

If you make money from this strategy, remember: (1) Correlation isn't causation, (2) Markets can stay irrational longer than you can stay solvent, (3) The authors accepts donations.

Also, maybe vote for campaign finance reform. Or at least politicians who support stricter trading rules for elected officials. Democracy shouldn't come with a brokerage account.

---
