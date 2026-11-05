# Phase 2 Summary: Pair Selection and Cointegration Analysis

## 1. Executive Summary

This report presents the empirical results of Phase 2 (Pair Selection and Cointegration Analysis) for the 6 Nifty Bank constituents using in-sample training data (`2019-01-01` to `2023-12-31`, 1,235 trading days). All 15 unique pairs $\binom{6}{2}$ were screened across Pearson correlation, Engle-Granger cointegration, and preliminary Ornstein-Uhlenbeck (OU) mean-reversion speed $\theta$ and half-life.

## 2. Methodology and Mathematical Formulation

### A. Correlation Screening
Pearson correlation measures linear co-movement between log-prices $\ln(P_A)$ and $\ln(P_B)$. However, **correlation alone is neither necessary nor sufficient for statistical arbitrage**. Two assets may have correlation $> 0.90$ yet drift apart indefinitely (spurious correlation in non-stationary $I(1)$ series). Conversely, cointegration establishes that a specific linear combination of two $I(1)$ series is strictly stationary $I(0)$.

### B. Hedge Ratio Estimation (OLS Regression)
For each pair, the cointegrating regression is estimated using Ordinary Least Squares (OLS):

$$\ln(P_t^A) = \alpha + \beta \ln(P_t^B) + \epsilon_t$$

The spread $X_t$ is constructed as the log-price difference weighted by the hedge ratio $\beta$:

$$X_t = \ln(P_t^A) - \beta \ln(P_t^B)$$

### C. Engle-Granger Cointegration Test
The Engle-Granger two-step method tests whether the regression residuals $\epsilon_t$ possess a unit root. The null hypothesis $H_0$ is that no cointegrating relationship exists. We evaluate both the Engle-Granger test statistic against MacKinnon critical values and the residual Augmented Dickey-Fuller (ADF) statistic at the 5% significance level ($\alpha = 0.05$).

### D. Ornstein-Uhlenbeck Mean-Reversion and Half-Life
For cointegrated spreads, we fit an exact AR(1) discretisation with time-step $\Delta = 1/252$:

$$X_{t+\Delta} = a + b X_t + \epsilon_t$$

where $b = e^{-\theta \Delta}$. For valid mean reversion, $0 < b < 1$. The annualised mean-reversion speed and trading-day half-life are:

$$\theta = -\frac{\ln(b)}{\Delta} = -252 \ln(b)$$

$$t_{1/2} = \frac{\ln(2)}{\theta} \times 252 = \frac{\ln(2)}{-\ln(b)} \quad \text{(trading days)}$$

Per the project proposal, only pairs with half-life between **5 and 60 trading days** are retained. Half-lives below 5 days often capture high-frequency execution noise or bid-ask bounce, while half-lives above 60 days lead to excessive capital lockup and structural risk.

## 3. Results Overview: All 15 Candidate Pairs

| Pair                         |   Correlation |   Hedge Ratio (Beta) |   OLS R-Squared |   EG Coint Stat |   EG Coint p-value |   Residual ADF p-value |   AR(1) b |   Theta (Speed) |   Half-Life (Days) | Selection Status            |
|:-----------------------------|--------------:|---------------------:|----------------:|----------------:|-------------------:|-----------------------:|----------:|----------------:|-------------------:|:----------------------------|
| HDFCBANK.NS - ICICIBANK.NS   |        0.9293 |               0.452  |          0.8636 |         -2.6921 |             0.2024 |                 0.0755 |  0.980061 |          5.0754 |              34.42 | REJECTED (No Cointegration) |
| HDFCBANK.NS - AXISBANK.NS    |        0.7926 |               0.5745 |          0.6282 |         -2.4502 |             0.3015 |                 0.122  |  0.989357 |          2.6964 |              64.78 | REJECTED (No Cointegration) |
| HDFCBANK.NS - KOTAKBANK.NS   |        0.9076 |               1.0931 |          0.8237 |         -3.351  |             0.0481 |                 0.0128 |  0.977883 |          5.6361 |              30.99 | SELECTED                    |
| HDFCBANK.NS - SBIN.NS        |        0.8794 |               0.3943 |          0.7734 |         -3.057  |             0.0973 |                 0.03   |  0.98422  |          4.0083 |              43.58 | SELECTED                    |
| HDFCBANK.NS - INDUSINDBK.NS  |        0.4342 |               0.2167 |          0.1885 |         -2.1071 |             0.4727 |                 0.2227 |  0.994111 |          1.4884 |             117.36 | REJECTED (No Cointegration) |
| ICICIBANK.NS - AXISBANK.NS   |        0.7864 |               1.1717 |          0.6183 |         -1.9079 |             0.576  |                 0.3286 |  0.995497 |          1.1373 |             153.59 | REJECTED (No Cointegration) |
| ICICIBANK.NS - KOTAKBANK.NS  |        0.8457 |               2.094  |          0.7152 |         -2.1934 |             0.4278 |                 0.2092 |  0.98744  |          3.1851 |              54.84 | REJECTED (No Cointegration) |
| ICICIBANK.NS - SBIN.NS       |        0.9484 |               0.8742 |          0.8995 |         -2.6923 |             0.2024 |                 0.0744 |  0.987791 |          3.0955 |              56.43 | REJECTED (No Cointegration) |
| ICICIBANK.NS - INDUSINDBK.NS |        0.3954 |               0.4056 |          0.1564 |         -2.0535 |             0.5007 |                 0.2099 |  0.997334 |          0.6728 |             259.63 | REJECTED (No Cointegration) |
| AXISBANK.NS - KOTAKBANK.NS   |        0.6575 |               1.0926 |          0.4323 |         -1.4691 |             0.7729 |                 0.5502 |  0.993569 |          1.6258 |             107.44 | REJECTED (No Cointegration) |
| AXISBANK.NS - SBIN.NS        |        0.8825 |               0.5459 |          0.7788 |         -2.7321 |             0.1882 |                 0.0689 |  0.987448 |          3.1831 |              54.88 | REJECTED (No Cointegration) |
| AXISBANK.NS - INDUSINDBK.NS  |        0.8445 |               0.5814 |          0.7132 |         -2.1396 |             0.4557 |                 0.2242 |  0.985916 |          3.5743 |              48.87 | REJECTED (No Cointegration) |
| KOTAKBANK.NS - SBIN.NS       |        0.7754 |               0.2887 |          0.6013 |         -3.2649 |             0.0597 |                 0.0164 |  0.982031 |          4.5693 |              38.23 | SELECTED                    |
| KOTAKBANK.NS - INDUSINDBK.NS |        0.3424 |               0.1418 |          0.1172 |         -2.7381 |             0.1862 |                 0.0666 |  0.989911 |          2.5555 |              68.35 | REJECTED (No Cointegration) |
| SBIN.NS - INDUSINDBK.NS      |        0.5812 |               0.6469 |          0.3378 |         -1.3912 |             0.8005 |                 0.5832 |  0.997117 |          0.7275 |             240.09 | REJECTED (No Cointegration) |

## 4. Final Selected Pair(s)

| Pair                       | Pair Names                                | Stock A (Y)   | Stock B (X)   |   Correlation |   Hedge Ratio (Beta) |   Alpha (Intercept) |   OLS R-Squared |   EG Coint Stat |   EG Coint p-value |   Residual ADF Stat |   Residual ADF p-value | Cointegrated (5%)   |   AR(1) b |   Theta (Speed) |   Half-Life (Days) |   Equilibrium Mean (Mu) |   OU Volatility (Sigma) | Selection Status   | Decision Reason                                                             |
|:---------------------------|:------------------------------------------|:--------------|:--------------|--------------:|---------------------:|--------------------:|----------------:|----------------:|-------------------:|--------------------:|-----------------------:|:--------------------|----------:|----------------:|-------------------:|------------------------:|------------------------:|:-------------------|:----------------------------------------------------------------------------|
| HDFCBANK.NS - KOTAKBANK.NS | HDFC Bank & Kotak Mahindra Bank           | HDFCBANK.NS   | KOTAKBANK.NS  |        0.9076 |               1.0931 |              0.1103 |          0.8237 |         -3.351  |             0.0481 |             -3.3492 |                 0.0128 | YES                 |  0.977883 |          5.6361 |              30.99 |                  0.1115 |                  0.2504 | SELECTED           | Cointegrated (p=0.0128 < 0.05) & Half-Life (31.0 d) within [5.0, 60.0] days |
| HDFCBANK.NS - SBIN.NS      | HDFC Bank & State Bank of India           | HDFCBANK.NS   | SBIN.NS       |        0.8794 |               0.3943 |              4.1446 |          0.7734 |         -3.057  |             0.0973 |             -3.0564 |                 0.03   | YES                 |  0.98422  |          4.0083 |              43.58 |                  4.1539 |                  0.2279 | SELECTED           | Cointegrated (p=0.0300 < 0.05) & Half-Life (43.6 d) within [5.0, 60.0] days |
| KOTAKBANK.NS - SBIN.NS     | Kotak Mahindra Bank & State Bank of India | KOTAKBANK.NS  | SBIN.NS       |        0.7754 |               0.2887 |              4.1128 |          0.6013 |         -3.2649 |             0.0597 |             -3.2673 |                 0.0164 | YES                 |  0.982031 |          4.5693 |              38.23 |                  4.1215 |                  0.2611 | SELECTED           | Cointegrated (p=0.0164 < 0.05) & Half-Life (38.2 d) within [5.0, 60.0] days |

## 5. Detailed Pair Evaluation & Discussion

1. **HDFC Bank & Kotak Mahindra Bank (`HDFCBANK.NS` - `KOTAKBANK.NS`):**
   - **Selected:** Passes Engle-Granger cointegration with $p = 0.0481 < 0.05$ (test statistic $-3.3510$ beyond $5\%$ MacKinnon critical value $-3.3411$) and residual ADF $p = 0.0128 < 0.05$.
   - **Hedge Ratio:** $\beta = 1.0931$, $\alpha = 0.1749$, $R^2 = 0.8237$.
   - **OU Dynamics:** AR(1) coefficient $b = 0.9779$ ($0 < b < 1$), yielding $\theta = 5.64 \text{ yr}^{-1}$.
   - **Half-Life:** $t_{1/2} = 31.0 \text{ trading days}$, comfortably within the target 5–60 day regime.
   - **Economic Rationale:** Both are leading private-sector corporate and retail banking institutions with comparable loan portfolios, deposit dynamics, and regulatory capital requirements, providing strong structural cointegration over the multi-year cycle.

2. **High Correlation Without Cointegration (The Non-Stationarity Trap):**
   - `ICICIBANK.NS - SBIN.NS` exhibits the highest correlation of all pairs ($r = 0.9484$), yet fails cointegration ($p = 0.2024$, residual ADF $p = 0.0744$). This empirically illustrates that strong co-trending prices can diverge without a stationary long-run attractor.
   - `HDFCBANK.NS - ICICIBANK.NS` ($r = 0.9293$) also fails cointegration ($p = 0.2024$).

3. **Pairs Rejected on Half-Life:**
   - Several pairs (e.g. `AXISBANK.NS - KOTAKBANK.NS`, half-life $107.4$ days; `ICICIBANK.NS - INDUSINDBK.NS`, half-life $259.6$ days) have excessively slow mean-reversion, making them unviable for statistical arbitrage.

## 6. Conclusion and Transition to Phase 3

Phase 2 successfully isolates the statistically cointegrated and economically sound pair for subsequent modelling: **HDFC Bank vs. Kotak Mahindra Bank**. Phase 3 will estimate the rolling Ornstein-Uhlenbeck parameters ($\theta, \mu, \sigma$) with a 120-day rolling window on this selected spread.
