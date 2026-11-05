# Phase 1 Independent Validation Report

## Overview

This report provides an independent audit of the completed Phase 1 Data Collection and Preprocessing for the OU Pairs Trading project. All datasets, raw downloads, price transformations, and train/test splits were verified without modifying underlying files.

## Validation Checklist (10 Criteria)

| Criterion | Status | Verification Details |
|---|:---:|---|
| 1. All six target stocks present | **PASS** | Present tickers (6): ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'INDUSINDBK.NS']. Missing: [] |
| 2. Adjusted closing prices used (not raw close) | **PASS** | Matches raw Adj Close: True; Differs from unadjusted Close: True |
| 3. Dates chronological, unique & aligned | **PASS** | Monotonically increasing: True, Duplicates: 0, Unaligned dates: 0 |
| 4. No missing, zero, negative or invalid prices | **PASS** | Total NaNs: 0, Values <= 0: 0, Infs: 0 |
| 5. Log prices equal natural logarithm ln(P_t) | **PASS** | Max absolute discrepancy across all 1,915 x 6 values: 1.78e-15 |
| 6. Training data range (2019-01-01 to 2023-12-31 only) | **PASS** | Train first date: 2019-01-01, last date: 2023-12-29 (1235 rows) |
| 7. Testing data range (2024-01-01 to 2026-09-30 only) | **PASS** | Test first date: 2024-01-01, last date: 2026-09-30 (680 rows) |
| 8. Zero overlap between training and testing sets | **PASS** | Overlapping dates count: 0 |
| 9. Raw data preserved unchanged | **PASS** | raw_adj_close.csv (225,816 bytes), raw_market_data.csv (984,908 bytes) |
| 10. Observation count matches proposal (1,915 days) | **PASS** | Cleaned: 1915 days. Train: 1235 + Test: 680 = 1915. Exact match to proposal. |

## Dataset Inventory & Observation Counts

| Dataset              |   Rows |   Columns | First Date   | Last Date   |   Nulls |
|:---------------------|-------:|----------:|:-------------|:------------|--------:|
| raw_adj_close.csv    |   1915 |         6 | 2019-01-01   | 2026-09-30  |       0 |
| cleaned_prices.csv   |   1915 |         6 | 2019-01-01   | 2026-09-30  |       0 |
| log_prices.csv       |   1915 |         6 | 2019-01-01   | 2026-09-30  |       0 |
| train_prices.csv     |   1235 |         6 | 2019-01-01   | 2023-12-29  |       0 |
| test_prices.csv      |    680 |         6 | 2024-01-01   | 2026-09-30  |       0 |
| train_log_prices.csv |   1235 |         6 | 2019-01-01   | 2023-12-29  |       0 |
| test_log_prices.csv  |    680 |         6 | 2024-01-01   | 2026-09-30  |       0 |

## Numerical Log-Price Verification Samples

Five randomly sampled price observations were verified using high-precision calculation:

| Date       | Ticker       |    Price |   Dataset Log-Price |   Computed ln(P) |   Abs Difference |
|:-----------|:-------------|---------:|--------------------:|-----------------:|-----------------:|
| 2024-03-21 | AXISBANK.NS  | 1033.12  |             6.94034 |          6.94034 |                0 |
| 2020-01-24 | AXISBANK.NS  |  733.768 |             6.59819 |          6.59819 |                0 |
| 2023-05-24 | SBIN.NS      |  541.342 |             6.29405 |          6.29405 |                0 |
| 2020-06-05 | HDFCBANK.NS  |  481.42  |             6.17674 |          6.17674 |                0 |
| 2026-02-11 | ICICIBANK.NS | 1394.35  |             7.24018 |          7.24018 |                0 |

## Discrepancies and Observations

- **Observation Count Consistency:** The cleaned and aligned dataset contains exactly **1,915 trading days** spanning `2019-01-01` to `2026-09-30`. This matches the baseline count established in the proposal with 0 discrepancy.
- **Data Integrity:** Zero nulls, zero duplicates, and zero non-positive values were detected. All 6 stocks traded on identical dates, requiring no imputation or forward-filling.
- **Temporal Partitioning:** Training (`2019-01-01` to `2023-12-29`, 1,235 observations) and Testing (`2024-01-01` to `2026-09-30`, 680 observations) have zero overlap and partition the full period completely.
- **Workspace Organization:** The root directory is completely clean with all Phase 1 datasets, scripts, tests, and figures isolated inside `Phase_1_Data_Preprocessing/`.
- **Validation Verdict:** **ALL 10 CHECKS PASSED**. Phase 1 dataset outputs are verified and ready for Phase 2.
