# Phase 1 Data Quality and Preprocessing Report

- **Data Universe:** 6 Nifty Bank Constituents
- **Date Range:** 2019-01-01 to 2026-09-30
- **Total Clean Observations:** 1915 trading days
- **Training Period (2019-01-01 to 2023-12-31):** 1235 trading days
- **Testing Period (2024-01-01 to 2026-09-30):** 680 trading days
- **Total Missing Observations:** 0
- **Duplicate Dates:** 0

## Asset Summary Statistics

| Ticker        | Asset Name          |   Total Observations |   Train Obs (2019-2023) |   Test Obs (2024-2026) |   Min Price (INR) |   Max Price (INR) |   Mean Price (INR) |   Std Price (INR) |   Min Log-Price |   Max Log-Price |   Mean Log-Price |   Annualised Return (%) |   Annualised Volatility (%) |
|:--------------|:--------------------|---------------------:|------------------------:|-----------------------:|------------------:|------------------:|-------------------:|------------------:|----------------:|----------------:|-----------------:|------------------------:|----------------------------:|
| HDFCBANK.NS   | HDFC Bank           |                 1915 |                    1235 |                    680 |            357.66 |            996.42 |             713.41 |            137.73 |          5.8796 |          6.9042 |           6.5503 |                    7.79 |                       24.86 |
| ICICIBANK.NS  | ICICI Bank          |                 1915 |                    1235 |                    680 |            272.38 |           1464.85 |             854    |            363.38 |          5.6072 |          7.2895 |           6.646  |                   21.74 |                       28.81 |
| AXISBANK.NS   | Axis Bank           |                 1915 |                    1235 |                    680 |            301.66 |           1401.92 |             892.03 |            256.73 |          5.7093 |          7.2456 |           6.7478 |                   14.4  |                       32.9  |
| KOTAKBANK.NS  | Kotak Mahindra Bank |                 1915 |                    1235 |                    680 |            218.36 |            452.46 |             353.2  |             48.12 |          5.3861 |          6.1147 |           5.8569 |                   10.51 |                       27.14 |
| SBIN.NS       | State Bank of India |                 1915 |                    1235 |                    680 |            136.73 |           1206.06 |             544.87 |            263.46 |          4.918  |          7.0951 |           6.17   |                   21.46 |                       31.06 |
| INDUSINDBK.NS | IndusInd Bank       |                 1915 |                    1235 |                    680 |            290.5  |           1746.57 |            1063    |            302.54 |          5.6716 |          7.4654 |           6.9223 |                    3.56 |                       45.93 |

## Data Validation Verification

1. **Zero / Negative Prices:** None detected across all series.
2. **Missing Observations:** All 6 constituent stocks traded on all 1,915 days with zero missing adjusted close values.
3. **Common Calendar Alignment:** Verified complete date overlap; no imputation or forward-filling was performed.
4. **Log Transformation:** Verified natural logarithms ln(P_t) are finite and well-defined.
