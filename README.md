# Stochastic Modelling of Statistical Arbitrage Using Ornstein–Uhlenbeck Processes: A Mean-Reversion Based Pairs Trading Strategy

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Institution:** BITS Pilani

---

## Phased Project Structure

The project is structured modularly by implementation phase to maintain clean separation between research stages:

```text
ou-mean-reversion-statistical-arbitrage/
├── Phase_1_Data_Preprocessing/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── raw_market_data.csv         # Raw multi-index OHLCV data from yfinance
│   │   │   └── raw_adj_close.csv           # Raw daily adjusted closing prices
│   │   └── processed/
│   │       ├── cleaned_prices.csv          # Aligned daily adjusted prices (1,915 obs)
│   │       ├── log_prices.csv              # Natural log-prices ln(P_t)
│   │       ├── train_prices.csv            # In-sample train prices (2019-01-01 to 2023-12-31, 1,235 obs)
│   │       ├── test_prices.csv             # Out-of-sample test prices (2024-01-01 to 2026-09-30, 680 obs)
│   │       ├── train_log_prices.csv        # In-sample train log-prices
│   │       └── test_log_prices.csv         # Out-of-sample test log-prices
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── data_loader.py                  # Yahoo Finance retrieval and caching
│   │   ├── preprocessing.py                # Validation, log transformation, splits & plotting
│   │   ├── run_phase1.py                   # Phase 1 pipeline execution entry point
│   │   └── validate_phase1.py              # Independent Phase 1 validation script
│   ├── results/
│   │   ├── figures/
│   │   │   ├── asset_prices.png            # Daily adjusted closing prices with split line
│   │   │   ├── normalized_asset_prices.png # Normalized trajectories (Base = 100)
│   │   │   └── log_prices.png              # Natural log-prices comparison
│   │   ├── tables/
│   │   │   ├── data_quality_summary.csv    # Statistical & data-quality summary
│   │   │   └── data_quality_report.md      # Data quality markdown report
│   │   └── phase_1_validation_report.md    # 10-point independent validation audit report
│   └── tests/
│       └── test_phase1.py                  # Unit tests for Phase 1 functions
├── Phase_2_Pair_Selection/                 # (Upcoming Phase 2)
├── Phase_3_OU_Estimation/                  # (Upcoming Phase 3)
├── requirements.txt                        # Project dependencies
├── Implementation_Plan_OU_Pairs_Trading.md # Project implementation plan
└── README.md                               # Project documentation
```

---

## Phase 1: Data Collection & Preprocessing

### Target Assets (Nifty Bank Constituents)
- **HDFC Bank** (`HDFCBANK.NS`)
- **ICICI Bank** (`ICICIBANK.NS`)
- **Axis Bank** (`AXISBANK.NS`)
- **Kotak Mahindra Bank** (`KOTAKBANK.NS`)
- **State Bank of India** (`SBIN.NS`)
- **IndusInd Bank** (`INDUSINDBK.NS`)

### Sampling & Partitioning
- **Total Period:** 1 January 2019 to 30 September 2026 (1,915 trading days)
- **In-Sample Training Period:** 1 January 2019 to 31 December 2023 (1,235 trading days)
- **Out-of-Sample Testing Period:** 1 January 2024 to 30 September 2026 (680 trading days)
- **Data Quality:** 0 missing values, 0 duplicate dates, 0 non-positive prices; no forward-filling required.

---

## Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Phase 1 Pipeline
```bash
# Execute using cached raw data:
python Phase_1_Data_Preprocessing/scripts/run_phase1.py

# Force re-download from Yahoo Finance:
python Phase_1_Data_Preprocessing/scripts/run_phase1.py --force-download
```

### 3. Run Independent Validation
```bash
python Phase_1_Data_Preprocessing/scripts/validate_phase1.py
```

### 4. Run Automated Unit Tests
```bash
python -m unittest Phase_1_Data_Preprocessing/tests/test_phase1.py
```
