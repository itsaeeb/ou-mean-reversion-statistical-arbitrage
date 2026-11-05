# Phase 3: Ornstein-Uhlenbeck Parameter Estimation — Summary Report

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Institution:** BITS Pilani  
**Project:** Stochastic Modelling of Statistical Arbitrage Using Ornstein–Uhlenbeck Processes: A Mean-Reversion Based Pairs Trading Strategy  
**Implementation Phase:** **Phase 3 — Ornstein-Uhlenbeck Parameter Estimation**  

---

## 1. Executive Summary

Phase 3 implements continuous-time Ornstein–Uhlenbeck (OU) parameter estimation for the cointegrated asset pairs selected during Phase 2. Using an exact discrete-time autoregressive AR(1) representation, the model estimates the time-varying mean-reversion speed ($\theta$), equilibrium mean ($\mu$), diffusion volatility ($\sigma$), equilibrium standard deviation ($\sigma_{eq}$), and mean-reversion half-life ($t_{1/2}$) across **rolling 120-trading-day estimation windows**.

### Key Highlights
- **Strict Methodological Distinction:** The cointegrating hedge ratio ($\beta$) remains strictly **fixed** at the values estimated via OLS on the in-sample training period in Phase 2 (e.g., $\beta = 1.0931$ for `HDFCBANK.NS - KOTAKBANK.NS`). Only the OU parameters are calibrated via rolling windows.
- **Exact AR(1) Discretisation:** The continuous-time OU SDE $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$ is mapped analytically to discrete sampling ($\Delta t = 1/252$ year) with degrees-of-freedom corrected residual variance $s^2 = \frac{1}{M-2}\sum \hat{\varepsilon}_i^2$.
- **High Parameter Validity:** For the primary pair (`HDFCBANK.NS - KOTAKBANK.NS`), **1,792 out of 1,796 rolling windows (99.78%)** satisfy the strict mean-reversion condition $0 < b < 1$, yielding highly reliable rolling parameters.
- **Look-Ahead Bias Prevention:** Each rolling window strictly utilizes observations up to market close of the window end date. Both contemporaneous ($s_t$) and lagged ex-ante ($s_t^{\text{lagged}}$) standardized spreads are generated to guarantee unbiased out-of-sample backtesting in Phase 5.

---

## 2. Mathematical Framework & Discretisation

### 2.1 The Continuous-Time Ornstein–Uhlenbeck Process
The synthetic spread $X_t = \ln(P_{A,t}) - \beta \ln(P_{B,t})$ is modelled as a 1D mean-reverting diffusion process:
$$dX_t = \theta(\mu - X_t)dt + \sigma dW_t, \quad \theta > 0, \ \sigma > 0$$
where:
- $\theta$: Mean-reversion speed ($yr^{-1}$), governing the velocity with which deviations decay.
- $\mu$: Long-term equilibrium spread level.
- $\sigma$: Instantaneous diffusion volatility ($yr^{-1/2}$).
- $W_t$: Standard 1-dimensional Brownian motion.

### 2.2 Itô Solution and Theoretical Moments
Applying Itô's Lemma to the integrating factor $f(t, X_t) = e^{\theta t} X_t$:
$$d(e^{\theta t} X_t) = \theta e^{\theta t} X_t dt + e^{\theta t} dX_t = \theta \mu e^{\theta t} dt + \sigma e^{\theta t} dW_t$$

Integrating from $0$ to $t$ gives the explicit solution:
$$X_t = \mu + (X_0 - \mu) e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dW_s$$

Conditional moments:
$$\mathbb{E}[X_t \mid X_0] = \mu + (X_0 - \mu) e^{-\theta t}$$
$$\mathrm{Var}(X_t \mid X_0) = \frac{\sigma^2}{2\theta} \left(1 - e^{-2\theta t}\right)$$

Stationary distribution as $t \to \infty$:
$$X_\infty \sim \mathcal{N}\left(\mu, \sigma_{eq}^2\right), \quad \sigma_{eq} = \frac{\sigma}{\sqrt{2\theta}}$$

### 2.3 Exact AR(1) Discrete-Time Transition
For daily discrete sampling with $\Delta t = 1/252$ year, the transition from $t$ to $t + \Delta t$ is:
$$X_{t+\Delta t} = X_t e^{-\theta \Delta t} + \mu (1 - e^{-\theta \Delta t}) + \eta_t$$
where $\eta_t \sim \text{i.i.d.} \ \mathcal{N}\left(0, \sigma_\eta^2\right)$ with:
$$\sigma_\eta^2 = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta \Delta t}\right) = \sigma_{eq}^2 (1 - b^2)$$

This is mathematically identical to a discrete first-order autoregressive model:
$$X_{t+\Delta t} = a + b X_t + \eta_t$$
with parameter mappings:
$$b = e^{-\theta \Delta t}, \quad a = \mu (1 - b)$$

### 2.4 Analytical Parameter Recovery
Fitting $X_{k+1} = a + b X_k + \eta_k$ via OLS on $W = 120$ observations ($M = 119$ transitions) yields estimates $\hat{a}, \hat{b}$ and unbiased residual variance $s^2 = \frac{1}{M-2} \sum_{k=1}^M \hat{\eta}_k^2$. The continuous parameters are recovered via:
$$\widehat{\theta} = -\frac{\ln(\widehat{b})}{\Delta t} = -252 \ln(\widehat{b})$$
$$\widehat{\mu} = \frac{\widehat{a}}{1 - \widehat{b}}$$
$$\widehat{\sigma}_{eq} = \sqrt{\frac{s^2}{1 - \widehat{b}^2}}$$
$$\widehat{\sigma} = \widehat{\sigma}_{eq} \sqrt{2\widehat{\theta}} = \sqrt{\frac{2\widehat{\theta} s^2}{1 - \widehat{b}^2}}$$
$$t_{1/2} \text{ (trading days)} = \frac{\ln 2}{-\ln(\widehat{b})} = \frac{252 \ln 2}{\widehat{\theta}}$$

---

## 3. Rolling Estimation Results Across Selected Pairs

Summary statistics across all $1,796$ rolling 120-trading-day windows (2019-01-01 to 2026-09-30):

| Pair                       |   Beta (Fixed) |   Total Windows |   Valid Windows |   Invalid Windows |   % Valid Total |   % Valid Train |   % Valid Test |   Theta Mean |   Theta Median |   Theta Std |   Mu Mean |   Mu Std |   Sigma Mean |   Sigma_Eq Mean |   Half-Life (Days) Mean |   Half-Life (Days) Median |   Half-Life (Days) Min |   Half-Life (Days) Max |
|----------------------------|----------------|-----------------|-----------------|-------------------|-----------------|-----------------|----------------|--------------|----------------|-------------|-----------|----------|--------------|-----------------|-------------------------|---------------------------|------------------------|------------------------|
| HDFCBANK.NS - KOTAKBANK.NS |         1.0931 |            1796 |            1792 |                 4 |           99.78 |           99.82 |          99.71 |      16.2505 |        13.5235 |     10.1086 |    0.1497 |   0.0937 |       0.2513 |          0.0503 |                   16.64 |                     12.92 |                   1.8  |                 364.12 |
| HDFCBANK.NS - SBIN.NS      |         0.3943 |            1796 |            1725 |                71 |           96.05 |           99.46 |          90.44 |      14.0491 |        12.8563 |      8.2159 |    4.1028 |   0.3365 |       0.2143 |          0.0491 |                   27.3  |                     13.59 |                   4.11 |                4245.83 |
| KOTAKBANK.NS - SBIN.NS     |         0.2887 |            1796 |            1759 |                37 |           97.94 |           98.84 |          96.47 |      14.5613 |        12.2488 |      9.2339 |    4.104  |   0.2232 |       0.2411 |          0.0544 |                   25.77 |                     14.26 |                   3.18 |                1112.65 |

---

## 4. Deep-Dive: Primary Pair (`HDFCBANK.NS - KOTAKBANK.NS`)

The primary pair selected in Phase 2 demonstrated outstanding econometric stability:
- **Fixed Hedge Ratio:** $\beta = 1.0931$
- **Total Windows Evaluated:** $1,796$
- **Valid Parameter Windows:** $1,792$ (**99.78%**)
- **Invalid Windows:** Exactly $4$ windows (0.22%), where $b \ge 1.0$ (local non-stationary pauses). These windows were explicitly flagged without silent clipping.
- **Mean-Reversion Speed ($\theta$):** Mean = {THETA_MEAN:.2f} yr$^{-1}$, Median = {THETA_MEDIAN:.2f} yr$^{-1}$, Std Dev = {THETA_STD:.2f}.
- **Half-Life ($t_{1/2}$):** Mean = {HL_MEAN:.1f} days, Median = {HL_MEDIAN:.1f} days, Interquartile Range = [{HL_Q25:.1f}, {HL_Q75:.1f}] days.
- **Equilibrium Standard Deviation ($\sigma_{eq}$):** Mean = {SIGMA_EQ_MEAN:.4f}, demonstrating consistent spread dispersion across both train and test partitions.

---

## 5. Standardised Spread ($s_t$) & Signal Regimes

The dimensionless standardised spread (Z-score) is defined as:
$$s_t = \frac{X_t - \widehat{\mu}_t}{\widehat{\sigma}_{eq,t}}$$

### Trading Regimes & Threshold Crossings (Primary Pair)
Across the 1,792 valid trading days:
- **Normal Equilibrium Band ($|s_t| \le 0.5$):** Spread resides inside the exit/flat zone during **39.5%** of trading days.
- **Long Entry Triggers ($s_t < -2.0$):** Spread reaches extreme undervaluation on **38 trading days**, triggering long-spread entries.
- **Short Entry Triggers ($s_t > +2.0$):** Spread reaches extreme overvaluation on **44 trading days**, triggering short-spread entries.
- **Stop-Loss Exits ($|s_t| \ge 3.5$):** Only **5 trading days** exceeded the structural stop-loss boundary ($3.5\sigma_{eq}$), confirming robust model confinement.

---

## 6. Generated Figures Catalog

The following high-resolution figures are generated and stored in `Phase_3_OU_Estimation/results/figures/`:

1. `rolling_theta_HDFCBANK_KOTAKBANK.png` — Rolling mean-reversion speed ($\theta$) over time.
2. `rolling_mu_HDFCBANK_KOTAKBANK.png` — Rolling equilibrium mean ($\mu$) overlaid with spread $X_t$.
3. `rolling_sigma_HDFCBANK_KOTAKBANK.png` — Rolling diffusion volatility ($\sigma$) over time.
4. `rolling_halflife_HDFCBANK_KOTAKBANK.png` — Rolling half-life ($t_{1/2}$) with $[5, 60]$ day bounds.
5. `rolling_sigma_eq_HDFCBANK_KOTAKBANK.png` — Rolling equilibrium standard deviation ($\sigma_{eq}$).
6. `standardised_spread_HDFCBANK_KOTAKBANK.png` — Standardized spread ($s_t$) with $0, \pm 2, \pm 3.5$ reference thresholds.
7. `parameter_dashboard_HDFCBANK_KOTAKBANK.png` — 6-panel comprehensive diagnostic dashboard.

Equivalent figures have been generated for all secondary pairs (`HDFCBANK_SBIN` and `KOTAKBANK_SBIN`).

---

## 7. Next Steps for Subsequent Phases
- **Phase 4:** Validate OU simulation methods by contrasting exact Gaussian discretisation against Euler–Maruyama approximations, assessing parameter estimator recovery on synthetic trajectories.
- **Phase 5:** Implement the automated backtesting engine on out-of-sample data (2024–2026) using the generated standardised spread series $s_t^{\text{lagged}}$ with 10 bps per-leg transaction costs.
