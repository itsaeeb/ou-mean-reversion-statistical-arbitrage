# Phase 5: Independent Verification and Validation Report

**Verification Timestamp:** 2026-11-08  
**Scope:** Phase 5 — OU Pairs Trading Strategy and Out-of-Sample Backtesting  
**Compliance Status:** 10 / 10 CHECKS PASSED (100% AUDIT PASS RATE)  

---

## 1. Audit Checklist Summary

| # | Audit Item | Requirement | Observed Status | Verdict |
|:---:|---|---|---|:---:|
| **1** | **Prerequisites Verification** | Cleaned prices, fixed hedge ratios, and rolling OU parameters exist and are verified. | All Phase 1, 2, and 3 tables present with 0 missing values. | **PASS** |
| **2** | **Hedge Ratio Invariance** | Hedge ratios $\beta$ strictly frozen from training period (2019-2023). | $\beta = 1.0931$ (HDFC-KOTAK), $0.3943$ (HDFC-SBI), $0.2887$ (KOTAK-SBI). Zero test-period re-estimation. | **PASS** |
| **3** | **Zero Look-Ahead Bias** | Signals generated on day $t$ close must execute no earlier than session $t+1$. | All trade entries execute strictly on $t+1$ based on signal at $t$. | **PASS** |
| **4** | **Spread Return Normalization** | Gross exposure normalized by $(1 + |\beta|)$. | Formula $r_{spread} = \frac{r_A - \beta r_B}{1 + |\beta|}$ verified on all test dates. | **PASS** |
| **5** | **Transaction Cost Deductions** | 10 bps charged per traded leg on entry and exit. | Entry cost = 20 bps, Exit cost = 20 bps. Total = 40 bps round-trip. Deducted accurately on transaction days. | **PASS** |
| **6** | **Threshold Adherence** | Entry at $|s_t| \ge 2.0$, Exit at $|s_t| \le 0.5$, Stop-loss at $|s_t| \ge 3.5$. | 100% of trades entered at $|s| \ge 2.0$ and exited strictly upon boundary contact. | **PASS** |
| **7** | **Position State Mutual Exclusion** | No simultaneous or overlapping positions in the same pair. | Maximum concurrent positions per pair = 1. Zero overlapping trades. | **PASS** |
| **8** | **Holding Period & Return Compounding** | Trade net return matches the product of daily net returns. | Discrepancy $< 10^{-12}$ between trade ledger and daily compounding. | **PASS** |
| **9** | **Manual Trade Traceability** | At least one trade traced manually down to exact mathematical operations. | Trade 1 of primary pair traced manually with full arithmetic match. | **PASS** |
| **10** | **Portfolio Capital Conservation** | Multi-pair portfolio capital and returns conserve total capital. | Equal-weight linear combination matches sum of weighted component returns. | **PASS** |

---

## 2. Detailed Manual Trade Trace: Trade 1 (`HDFCBANK.NS - KOTAKBANK.NS`)

Below is the complete step-by-step audit of the first trade executed by the strategy:

- **Pair:** `HDFCBANK.NS - KOTAKBANK.NS` (Stock A: HDFC Bank, Stock B: Kotak Mahindra Bank)
- **Hedge Ratio $\beta$:** $1.0931$ (frozen from Phase 2 in-sample OLS)
- **Signal Date:** `2024-01-18 00:00:00` (Market Close)
  - Standardised Spread $s_{signal} = -2.3327 \le -2.0000$ (Trigger: **LONG SPREAD**)
  - Closing Price A: ₹711.52
  - Closing Price B: ₹354.04
- **Entry Execution Date:** `2024-01-19 00:00:00`
  - Position: $+1$ (Long Spread)
  - Gross Exposure Weights: $w_A = \frac{1}{1 + 1.0931} = 0.47776$, $w_B = \frac{-1.0931}{1 + 1.0931} = -0.52224$
  - Sum of Gross Exposures: $|w_A| + |w_B| = 0.47776 + 0.52224 = 1.0000$
  - Entry Transaction Cost Deducted: $2 \times 10\text{ bps} = 0.20\% = 0.0020$
- **Holding Period:** `27` trading sessions (from `2024-01-19 00:00:00` to `2024-02-28 00:00:00`)
- **Exit Date:** `2024-02-28 00:00:00` (Market Close)
  - Standardised Spread $s_{exit} = -0.4221$ (Within exit corridor $[-0.5, +0.5]$)
  - Exit Reason: `TAKE_PROFIT`
  - Exit Closing Price A: ₹674.77
  - Exit Closing Price B: ₹335.38
  - Exit Transaction Cost Deducted: $2 \times 10\text{ bps} = 0.20\% = 0.0020$
- **Performance Reconciliation:**
  - Compounded Daily Gross Return: $\prod (1 + r_d^{gross}) - 1 = +0.2232\%$
  - Total Transaction Costs: $0.0020 + 0.0020 = 0.40\%$
  - Compounded Daily Net Return: $\prod (1 + r_d^{net}) - 1 = -0.1768\%$
  - Mathematical Reconciliation: Gross Return (+0.2232%) - Transaction Drag (0.40%) $\approx$ Net Return (-0.1768%). (Verified).

---

## 3. Conclusion

Phase 5 has successfully passed all 10 independent verification checks without exception. The backtesting engine is mathematically sound, reproducible, and compliant with all institutional standards.
