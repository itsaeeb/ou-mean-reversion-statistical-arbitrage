# Phase 3: Ornstein-Uhlenbeck Parameter Estimation — Validation Report

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Institution:** BITS Pilani  
**Audit Date:** 2026-11-05  
**Audit Scope:** Independent mathematical, algorithmic, and statistical verification of Phase 3 rolling OU parameter estimation.

---

## 1. Audit Executive Summary

An independent 10-point audit was performed on all scripts, mathematical routines, output tables, and figures generated in Phase 3.

**Audit Result:** **10 out of 10 checks PASSED with 0 discrepancies.**

```text
Checks Evaluated: 10
Passed: 10
Failed: 0
Discrepancies: 0
Overall Status: VERIFIED & COMPLIANT
```

---

## 2. 10-Point Audit Results Table

| Check                                     | Requirement                                                                           | Observed                                                                      | Status   |
|-------------------------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|----------|
| 1. Fixed Hedge Ratio Consistency          | Beta matches Phase 2 in-sample OLS estimate and remains fixed across all windows      | All 3 pairs maintain fixed Beta                                               | PASSED   |
| 2. AR(1) OLS Regression Accuracy          | AR(1) coefficients and residual variance match independent linear algebra within 1e-9 | Max diff: 7.77e-16                                                            | PASSED   |
| 3. OU Parameter Recovery Formulas         | theta = -ln(b)/dt, mu = a/(1-b), sigma = sqrt(2*theta*s^2/(1-b^2)) exactly verified   | Theta diff: 0.0e+00, Mu diff: 0.0e+00, Sigma diff: 0.0e+00                    | PASSED   |
| 4. Sampling Frequency Delta t             | Delta t is strictly 1/252 year across all estimations                                 | Constant 1/252 (0.003968 yr) verified across 1792 windows                     | PASSED   |
| 5. Rolling Window Length                  | Every window must contain exactly 120 trading day observations                        | All 1796 windows have length = 120                                            | PASSED   |
| 6. Half-Life Unit Conversion              | Half-life in trading days = 252 * ln(2)/theta == ln(2)/(-ln(b))                       | Max conversion difference = 7.08e-12 days                                     | PASSED   |
| 7. Equilibrium Std Dev (sigma_eq)         | sigma_eq = sigma / sqrt(2*theta) == sqrt(s^2 / (1 - b^2)) mathematically verified     | Max difference = 3.21e-14                                                     | PASSED   |
| 8. Standardized Spread Alignment          | s_t = (X_t - mu_t)/sigma_eq,t evaluated at Window_End_Date with zero look-ahead       | Z-score exact match, Date = Window_End_Date verified across all 1,796 windows | PASSED   |
| 9. Explicit Validity Flagging             | b >= 1 or b <= 0 must set is_valid=False and parameter NaNs with no silent clipping   | 4 invalid windows properly flagged with documented failure reasons and NaNs   | PASSED   |
| 10. Manually Verifiable Numerical Example | Analytical benchmark test with known values (b=0.96, a=0.02, s^2=0.0004) verified     | Theta=10.29, Mu=0.50, Sigma_eq=0.0714, Half-Life=17.0d, Z=2.00                | PASSED   |

---

## 3. Detailed Audit Findings

### 3.1 Fixed Hedge Ratio (Beta) Audit
The project requirements strictly mandate that the hedge ratio $\beta$ estimated via OLS on the Phase 2 training period must remain **fixed** throughout Phase 3:
| Pair                       |   Phase 2 Beta |   Phase 3 Beta (Fixed) | Matches Exactly   |
|----------------------------|----------------|------------------------|-------------------|
| HDFCBANK.NS - KOTAKBANK.NS |         1.0931 |                 1.0931 | PASS              |
| HDFCBANK.NS - SBIN.NS      |         0.3943 |                 0.3943 | PASS              |
| KOTAKBANK.NS - SBIN.NS     |         0.2887 |                 0.2887 | PASS              |

All candidate pairs maintain identical, fixed hedge ratios across all $1,796$ rolling windows with zero modification or rolling re-estimation.

### 3.2 Exact Discrete AR(1) and Parameter Recovery Formulas
For continuous-time OU process $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$ sampled at $\Delta t = 1/252$:
- AR(1) representation: $X_{t+\Delta t} = a + b X_t + \eta_t$
- Slope: $b = e^{-\theta \Delta t} \implies \widehat{\theta} = -252 \ln(\widehat{b})$
- Intercept: $a = \mu(1 - b) \implies \widehat{\mu} = \frac{\widehat{a}}{1 - \widehat{b}}$
- Residual variance: $s^2 = \frac{1}{M-2} \sum \hat{\eta}_k^2$
- Volatility: $\widehat{\sigma} = \sqrt{\frac{2\widehat{\theta} s^2}{1 - \widehat{b}^2}}$
- Equilibrium standard deviation: $\sigma_{eq} = \frac{\sigma}{\sqrt{2\theta}} = \sqrt{\frac{s^2}{1 - \widehat{b}^2}}$
- Half-life: $t_{1/2} = \frac{\ln 2}{\theta \Delta t} = \frac{\ln 2}{-\ln(\widehat{b})}$

All formulas were tested against independent linear algebra routines and matched with a maximum difference $< 10^{-9}$.

### 3.3 Explicit Rejection of Non-Stationary Windows (Zero Silent Clipping)
For windows exhibiting $b \ge 1.0$ (unit-root or explosive drift) or $b \le 0.0$ (oscillatory AR(1)), the model explicitly records `Is_Valid = False`, documents the exact reason in `Failure_Reason`, and assigns `NaN` to $\theta, \mu, \sigma, \sigma_{eq}, t_{1/2}, s_t$. No parameter is silently clipped into acceptable bounds.

### 3.4 Zero Look-Ahead Bias & Alignment
The rolling estimation engine uses the following alignment convention:
- For window ending at date $t$, data utilized is strictly $[t - 119, \dots, t]$.
- Contemporaneous Z-score $s_t = (X_t - \mu_t) / \sigma_{eq,t}$ is derived from parameters known at market close on date $t$.
- Lagged Z-score $s_t^{\text{lagged}} = (X_t - \mu_{t-1}) / \sigma_{eq,t-1}$ is derived from parameters strictly prior to date $t$.
- Date indexing strictly equals `Window_End_Date` across all $1,796$ windows.

---

## 4. Manually Verifiable Numerical Benchmark

To allow manual verification of the implementation, a deterministic benchmark example was calculated:

**Input Parameters:**
- $b = 0.96$
- $a = 0.02$
- $s^2 = 0.0004$
- $\Delta t = 1/252$

**Hand-Calculated Derivations:**
1. Mean-reversion speed:
   $$\theta = -252 \times \ln(0.96) = -252 \times (-0.04082199) = 10.287143 \text{ yr}^{-1}$$
2. Long-term mean:
   $$\mu = \frac{0.02}{1 - 0.96} = \frac{0.02}{0.04} = 0.500000$$
3. Equilibrium standard deviation:
   $$\sigma_{eq} = \sqrt{\frac{0.0004}{1 - 0.96^2}} = \sqrt{\frac{0.0004}{0.0784}} = \sqrt{0.00510204} = 0.07142857$$
4. Instantaneous volatility:
   $$\sigma = \sigma_{eq} \sqrt{2 \theta} = 0.07142857 \times \sqrt{2 \times 10.287143} = 0.07142857 \times 4.5358885 = 0.32399205$$
5. Half-life (trading days):
   $$t_{1/2} = \frac{\ln 2}{-\ln(0.96)} = \frac{0.69314718}{0.04082199} = 16.9797 \text{ trading days}$$
6. Standardised spread for $X_t = 0.64285714$:
   $$s_t = \frac{0.64285714 - 0.500000}{0.07142857} = \frac{0.14285714}{0.07142857} = +2.000000$$

The implementation reproduces these exact values within numerical tolerance ($< 10^{-6}$).

---

## 5. Limitations & Practical Considerations
1. **Window Size Sensitivity:** A fixed window of $W = 120$ trading days ($pprox 6$ months) balances responsiveness against parameter estimation variance. In high-volatility regimes (e.g., March 2020 COVID shock), $\sigma$ responds rapidly, temporarily widening $\sigma_{eq}$.
2. **Local Unit-Root Windows:** In 4 out of 1,796 windows ($0.22\%$) for `HDFCBANK.NS - KOTAKBANK.NS`, local trends caused $b \ge 1.0$. The trading state machine in Phase 5 must hold positions or abstain from new entries during invalid windows, preserving strategy robustness.
