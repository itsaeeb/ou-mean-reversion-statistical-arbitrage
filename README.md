# Stochastic Modelling of Statistical Arbitrage Using Ornstein–Uhlenbeck Processes: A Mean-Reversion Based Pairs Trading Strategy

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Institution:** Birla Institute of Technology and Science (BITS), Pilani  
**Implementation Language:** Python 3.10+  
**Target Market:** National Stock Exchange of India (NSE) — Nifty Bank Constituents  

---

## 1. Project Overview & Objectives

Statistical arbitrage pairs trading exploits temporary divergences in the relative pricing of two economically linked assets. While classical pairs trading often relies on heuristics, this project grounds the strategy in **stochastic calculus** by modelling the continuous-time dynamics of the synthetic log-price spread using an **Ornstein–Uhlenbeck (OU) mean-reverting process**.

### Core Objectives
1. **Mathematical Rigour:** Formulate the OU continuous-time stochastic differential equation (SDE), solve it via Itô's Lemma, and derive its theoretical expectation, variance, stationary distribution, and half-life.
2. **Empirical Cointegration & Selection:** Formulate the spread via Ordinary Least Squares (OLS) hedge ratio estimation, evaluate stationarity using the Engle–Granger two-step test on historical training data, and filter candidate pairs by mean-reversion speed.
3. **Parameter Estimation & Discretisation:** Implement exact analytical AR(1) discretisation and recover continuous OU parameters ($\theta, \mu, \sigma, \sigma_{eq}, t_{1/2}$) across rolling 120-trading-day windows with strict validity checks ($0 < b < 1$) and zero look-ahead bias.
4. **Simulation & Numerical Validation:** Contrast exact Gaussian discretisation against Euler–Maruyama approximations, verify theoretical moment dynamics via large-scale Monte Carlo, and assess estimator bias on synthetic paths.
5. **Algorithmic Trading & Out-of-Sample Backtesting:** Design a disciplined Z-score trading strategy with entry, exit, stop-loss thresholds, and transaction frictions (10 bps per leg) evaluated strictly out-of-sample.
6. **Sensitivity & Performance Evaluation:** Analyze strategy risk-adjusted returns (Sharpe ratio, maximum drawdown, win rate) across variable threshold bands, window lengths, and execution frictions.

---

## 2. Mathematical Methodology

### 2.1 Asset Universe & Log-Price Spread
Given two non-stationary price series $P_t^A$ and $P_t^B$, we consider their natural logarithms:
$$y_t = \ln(P_t^A), \quad x_t = \ln(P_t^B)$$

The cointegrating hedge ratio $\beta$ and intercept $\alpha$ are estimated via OLS on in-sample training data:
$$y_t = \alpha + \beta x_t + \varepsilon_t$$

The synthetic spread $X_t$ is defined as the linear combination:
$$X_t = y_t - \beta x_t = \ln(P_t^A) - \beta \ln(P_t^B)$$

When $y_t$ and $x_t$ are cointegrated, the residual series $\hat{\varepsilon}_t = X_t - \alpha$ is stationary $I(0)$, even though the underlying log-price series are integrated of order 1, $I(1)$.

### 2.2 Continuous-Time Ornstein–Uhlenbeck Process
The spread $X_t$ is governed by the 1-dimensional continuous-time stochastic differential equation (SDE):
$$dX_t = \theta (\mu - X_t) dt + \sigma dW_t, \quad \theta > 0, \ \sigma > 0$$
where:
- $\theta$: Rate of mean-reversion (velocity of drift towards equilibrium).
- $\mu$: Long-run equilibrium mean of the spread.
- $\sigma$: Instantaneous diffusion coefficient (spread volatility).
- $W_t$: Standard 1-dimensional Brownian motion on filtered probability space $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t\ge 0}, \mathbb{P})$.

### 2.3 Analytical Properties via Itô's Lemma
Applying Itô's Lemma to the transformed process $f(t, X_t) = e^{\theta t} X_t$:
$$d(e^{\theta t} X_t) = \theta e^{\theta t} X_t dt + e^{\theta t} dX_t = \theta \mu e^{\theta t} dt + \sigma e^{\theta t} dW_t$$

Integrating from $0$ to $t$ yields the **explicit continuous-time solution**:
$$X_t = \mu + (X_0 - \mu) e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dW_s$$

From this closed-form solution, the conditional distribution of $X_t \mid X_0$ is Gaussian:
$$\mathbb{E}[X_t \mid X_0] = \mu + (X_0 - \mu) e^{-\theta t}$$
$$\mathrm{Var}(X_t \mid X_0) = \frac{\sigma^2}{2\theta} \left(1 - e^{-2\theta t}\right)$$

As $t \to \infty$, the process converges to its unique **stationary distribution**:
$$X_\infty \sim \mathcal{N}\left(\mu, \sigma_{eq}^2\right), \quad \sigma_{eq} = \frac{\sigma}{\sqrt{2\theta}}$$

### 2.4 Half-Life of Mean Reversion
The half-life $t_{1/2}$ is the expected time for the expected spread deviation $(X_t - \mu)$ to decay by $50\%$:
$$\mathbb{E}[X_{t_{1/2}} - \mu] = \frac{1}{2}(X_0 - \mu) \implies e^{-\theta t_{1/2}} = \frac{1}{2} \implies t_{1/2} = \frac{\ln 2}{\theta}$$

Under daily sampling with time increment $\Delta t = \frac{1}{252}$ year:
$$t_{1/2} \text{ (trading days)} = \frac{t_{1/2} \text{ (years)}}{\Delta t} = \frac{252 \ln 2}{\theta} = \frac{\ln 2}{-\ln b}$$

### 2.5 Exact AR(1) Discrete-Time Transition
Sampling the continuous-time OU process at discrete intervals $\Delta t$ yields an exact Gaussian Autoregressive AR(1) process:
$$X_{t+\Delta t} = a + b X_t + \eta_t, \quad \eta_t \sim \text{i.i.d.} \ \mathcal{N}\left(0, \sigma_\eta^2\right)$$
where matching theoretical drift and conditional variance yields:
$$b = e^{-\theta \Delta t}, \quad a = \mu (1 - b), \quad \sigma_\eta^2 = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta \Delta t}) = \sigma_{eq}^2 (1 - b^2)$$

Fitting the linear regression $X_t = a + b X_{t-1} + \eta_t$ via OLS allows exact recovery of the underlying continuous parameters:
$$\widehat{\theta} = -\frac{\ln(\widehat{b})}{\Delta t} = -252 \ln(\widehat{b})$$
$$\widehat{\mu} = \frac{\widehat{a}}{1 - \widehat{b}}$$
$$\widehat{\sigma}_{eq} = \sqrt{\frac{s^2}{1 - \widehat{b}^2}}, \quad s^2 = \frac{1}{M-2}\sum_{k=1}^M \hat{\eta}_k^2$$
$$\widehat{\sigma} = \widehat{\sigma}_{eq} \sqrt{2\widehat{\theta}} = \sqrt{\frac{2\widehat{\theta} s^2}{1 - \widehat{b}^2}}$$
*Numerical Constraint:* For stationary mean reversion, the discrete autoregressive coefficient must strictly satisfy $0 < b < 1$. Non-stationary estimates ($b \ge 1.0$ or $b \le 0.0$) are explicitly rejected and flagged with `NaN` parameters without silent clipping.

### 2.6 Euler–Maruyama Discretisation & Stability
The first-order Euler–Maruyama approximation is given by:
$$X_{t+\Delta t}^{EM} = X_t^{EM} + \theta(\mu - X_t^{EM})\Delta t + \sigma \sqrt{\Delta t} Z_t = (1 - \theta \Delta t)X_t^{EM} + \theta \mu \Delta t + \sigma \sqrt{\Delta t} Z_t$$
- Amplification factor: $g = 1 - \theta \Delta t$.
- Linear stability requirement: $|1 - \theta \Delta t| < 1 \iff \Delta t < \frac{2}{\theta}$.
- If $\Delta t \ge \frac{2}{\theta}$, Euler–Maruyama becomes **explosively unstable**, whereas exact discretisation remains unconditionally stable (A-stable) for all $\Delta t > 0$.

### 2.7 Standardised Spread (Z-Score) & Trading Strategy Logic
Normalising the continuous-time spread by its rolling equilibrium moments produces the dimensionless Z-score:
$$s_t = \frac{X_t - \hat{\mu}_t}{\hat{\sigma}_{eq, t}}, \quad \hat{\sigma}_{eq, t} = \frac{\hat{\sigma}_t}{\sqrt{2\hat{\theta}_t}}$$

#### Position Sizing & Capital Normalization
To ensure total gross invested capital equals $1.0$ ($|w_A| + |w_B| = 1.0$), portfolio weights are normalized by $(1 + |\beta|)$:
$$w_A = \frac{1}{1 + |\beta|}, \quad w_B = -\frac{\beta}{1 + |\beta|}$$
The normalized gross return of the long spread position is:
$$r_{\text{spread}, t} = \frac{r_{A, t} - \beta r_{B, t}}{1 + |\beta|}$$
*(The short spread position earns $-r_{\text{spread}, t}$. Negative or invalid $\beta$ values are handled explicitly).*

#### State Machine Trading Rules
The strategy implements a disciplined mean-reverting threshold policy:
- **Long Spread Entry ($-3.5 < s_t \le -2.0$):** Spread is abnormally depressed. Enter Long Spread ($w_A$ of Stock A, $w_B$ of Stock B).
- **Short Spread Entry ($+2.0 \le s_t < +3.5$):** Spread is abnormally elevated. Enter Short Spread ($-w_A$ of Stock A, $-w_B$ of Stock B).
- **Mean-Reversion Take-Profit Exit ($|s_t| \le 0.5$):** Spread has reverted into the equilibrium band. Liquidate both legs to cash.
- **Stop-Loss Risk Exit ($|s_t| \ge 3.5$):** Spread divergence exceeds structural boundaries. Liquidate position immediately to prevent catastrophic loss.
- **Strict Zero Look-Ahead Execution:** Signals generated at the market close of day $t$ execute for the next trading session $t+1$. Daily returns are earned during session $t+1$ based on price change from close $t$ to close $t+1$.
- **Transaction Costs:** Realistic friction of $10\text{ bps}$ ($0.10\%$) applied to each traded leg on both entry and exit ($20\text{ bps}$ on entry, $20\text{ bps}$ on exit = $40\text{ bps}$ round-trip).

---

## 3. Project Implementation Status

| Phase | Description | Scope / Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Data Collection & Preprocessing** | Download 6 NSE banking stocks (2019–2026), align trading calendars, compute log prices, generate train/test splits, validate 0 missing values. | **Completed** |
| **Phase 2** | **Pair Selection & Cointegration Analysis** | Evaluate all 15 candidate pairs on train data, OLS hedge ratios ($\beta$), Engle–Granger tests ($p < 0.05$), AR(1) OU recovery, half-life filter ($5 \le t_{1/2} \le 60$). | **Completed** |
| **Phase 3** | **OU Parameter Estimation** | Exact AR(1) mapping ($\Delta t = 1/252$), rolling 120-day calibration ($\theta, \mu, \sigma, \sigma_{eq}, t_{1/2}$), fixed $\beta$, validity flags ($0 < b < 1$), and Z-score spread generation. | **Completed** |
| **Phase 4** | **OU Simulation & Numerical Validation** | Exact vs. Euler–Maruyama discretisation, coupled Brownian paths, stability boundaries ($\Delta t < 2/\theta$), 10,000-path Monte Carlo moment checks, and Hurwicz bias analysis. | **Completed** |
| **Phase 5** | **Trading Strategy & Out-of-Sample Backtesting** | Out-of-sample backtesting (2024–2026, 680 days), zero look-ahead bias, 10 bps per-leg transaction costs, stop-loss protection, single-pair & equal-weighted multi-pair portfolio. | **Completed** |
| **Phase 6** | **Performance Metrics & Benchmark Comparison** | Continuous-time OU first-passage-time (FPT) optimal thresholds, multi-strategy out-of-sample benchmarking, 2D parameter heatmaps, cost frontier sweeps, 10-point audit (10/10 passed). | **Completed** |
| **Phase 7** | **Report, Presentation & Documentation** | Comprehensive academic project report (3–4 pages), slide deck for viva defense, documentation consolidation. | *Not Started* |

---

## 4. Actual Repository Structure & Module Descriptions

```text
ou-mean-reversion-statistical-arbitrage/
├── Phase_1_Data_Preprocessing/                 # Phase 1: Data acquisition and curation
│   ├── data/
│   │   ├── raw/
│   │   │   ├── raw_market_data.csv             # Raw multi-index OHLCV data from yfinance (1,915 dates)
│   │   │   └── raw_adj_close.csv               # Raw daily adjusted closing prices
│   │   └── processed/
│   │       ├── cleaned_prices.csv              # Aligned adjusted prices across all 6 stocks (1,915 obs)
│   │       ├── log_prices.csv                  # Natural logarithm prices ln(P_t) (1,915 obs)
│   │       ├── train_prices.csv                # In-sample train prices (2019-01-01 to 2023-12-31, 1,235 obs)
│   │       ├── test_prices.csv                 # Out-of-sample test prices (2024-01-01 to 2026-09-30, 680 obs)
│   │       ├── train_log_prices.csv            # In-sample log-prices (1,235 obs)
│   │       └── test_log_prices.csv             # Out-of-sample log-prices (680 obs)
│   ├── results/
│   │   ├── figures/
│   │   │   ├── asset_prices.png                # Daily adjusted closing prices with train/test boundary
│   │   │   ├── normalized_asset_prices.png     # Normalized price trajectories (Base = 100)
│   │   │   └── log_prices.png                  # Comparative log-price trajectories
│   │   ├── tables/
│   │   │   ├── data_quality_summary.csv        # Summary stats (mean, std, min, max, missing count)
│   │   │   └── data_quality_report.md          # Markdown data quality summary table
│   │   └── phase_1_validation_report.md        # 10-point independent validation audit report
│   ├── scripts/
│   │   ├── __init__.py                         # Python package marker
│   │   ├── data_loader.py                      # Yahoo Finance data fetcher with local caching
│   │   ├── preprocessing.py                    # Validation, log transformation, train/test splitting, plotting
│   │   ├── run_phase1.py                       # Pipeline runner for Phase 1
│   │   └── validate_phase1.py                  # Independent Phase 1 verification script
│   ├── tests/
│   │   └── test_phase1.py                      # Unit tests for Phase 1 data pipeline
│   └── run_phase1.py                           # Top-level launcher for Phase 1
├── Phase_2_Pair_Selection/                     # Phase 2: Statistical pair selection & cointegration
│   ├── results/
│   │   ├── figures/
│   │   │   ├── correlation_matrix.png          # 6x6 Pearson log-price correlation heatmap
│   │   │   ├── correlation_vs_cointegration.png # Scatter plot: Correlation vs Engle-Granger p-value
│   │   │   ├── cointegration_pvalues.png       # Ranked bar chart of Engle-Granger p-values vs 5% critical line
│   │   │   └── selected_pair_spreads.png       # In-sample spread movement for candidate pairs
│   │   ├── tables/
│   │   │   ├── all_pairs_analysis.csv          # Comprehensive statistical metrics for all 15 candidate pairs
│   │   │   ├── cointegrated_pairs.csv          # Filtered pairs passing Engle-Granger cointegration (p < 0.05)
│   │   │   └── final_selected_pairs.csv        # Final pair meeting cointegration & 5-60 day half-life criteria
│   │   ├── phase_2_summary.md                  # Comprehensive empirical and theoretical methodology report
│   │   └── phase_2_validation_report.md        # 7-point independent validation audit report
│   ├── scripts/
│   │   ├── __init__.py                         # Python package marker
│   │   ├── pair_selection.py                   # OLS estimation, Engle-Granger test, AR(1) OU estimation & plotting
│   │   ├── run_phase2.py                       # Pipeline runner for Phase 2
│   │   └── validate_phase2.py                  # Independent Phase 2 verification script
│   ├── tests/
│   │   └── test_phase2.py                      # Unit tests for Phase 2 mathematical calculations
│   └── run_phase2.py                           # Top-level launcher for Phase 2
├── Phase_3_OU_Estimation/                      # Phase 3: Rolling Ornstein-Uhlenbeck parameter estimation
│   ├── results/
│   │   ├── figures/
│   │   │   ├── parameter_dashboard_HDFCBANK_KOTAKBANK.png  # 6-panel comprehensive diagnostic dashboard
│   │   │   ├── rolling_theta_HDFCBANK_KOTAKBANK.png        # Rolling theta (reversion speed) over time
│   │   │   ├── rolling_mu_HDFCBANK_KOTAKBANK.png           # Rolling equilibrium mean mu overlaid with spread
│   │   │   ├── rolling_sigma_HDFCBANK_KOTAKBANK.png        # Rolling diffusion volatility sigma over time
│   │   │   ├── rolling_halflife_HDFCBANK_KOTAKBANK.png     # Rolling half-life with [5, 60] day bounds
│   │   │   ├── rolling_sigma_eq_HDFCBANK_KOTAKBANK.png     # Rolling equilibrium standard deviation
│   │   │   ├── standardised_spread_HDFCBANK_KOTAKBANK.png  # Z-score spread with 0, +/-2, +/-3.5 thresholds
│   │   │   └── [equivalent 7 figures generated for HDFCBANK_SBIN and KOTAKBANK_SBIN]
│   │   ├── tables/
│   │   │   ├── rolling_params_HDFCBANK_KOTAKBANK.csv       # Primary pair rolling estimates (1,796 windows)
│   │   │   ├── rolling_params_HDFCBANK_SBIN.csv            # Secondary pair rolling estimates
│   │   │   ├── rolling_params_KOTAKBANK_SBIN.csv           # Secondary pair rolling estimates
│   │   │   ├── combined_rolling_parameters.csv             # Consolidated parameter records across all pairs
│   │   │   ├── standardised_spreads.csv                    # Contemporaneous & lagged Z-score series
│   │   │   └── rolling_parameter_summary.csv               # Aggregate distribution moments & validity rates
│   │   ├── phase_3_summary.md                  # Comprehensive empirical and theoretical methodology report
│   │   └── phase_3_validation_report.md        # 10-point independent validation audit report (10/10 passed)
│   ├── scripts/
│   │   ├── __init__.py                         # Python package marker
│   │   ├── ou_estimation.py                    # Exact AR(1) OLS, parameter recovery, rolling engine, plotting
│   │   ├── run_phase3.py                       # Pipeline runner for Phase 3
│   │   └── validate_phase3.py                  # Independent Phase 3 verification script
│   ├── tests/
│   │   └── test_phase3.py                      # Unit tests for Phase 3 mathematical calculations
│   └── run_phase3.py                           # Top-level launcher for Phase 3
├── Phase_4_OU_Simulation/                      # Phase 4: OU simulation and numerical validation
│   ├── experiments/
│   │   ├── discretization_comparison.py        # Experiment 1: Exact vs Euler error scaling across dt & stability
│   │   ├── moment_validation.py                # Experiment 2: 10,000-path Monte Carlo moment validation vs theory
│   │   └── parameter_recovery.py               # Experiment 3: Synthetic parameter recovery & Hurwicz bias analysis
│   ├── results/
│   │   ├── figures/
│   │   │   ├── exact_ou_paths.png              # Exact OU path bundle with theoretical +/-2 sigma(t) envelope
│   │   │   ├── euler_maruyama_paths.png        # Euler-Maruyama simulated path bundle
│   │   │   ├── exact_vs_euler_paths.png        # 4-panel coupled comparison (Daily, Monthly, Quarterly, Unstable)
│   │   │   ├── simulation_error_vs_timestep.png# MAE and RMSE scaling monotonically with time step dt
│   │   │   ├── empirical_vs_theoretical_mean.png # Monte Carlo mean tracking theoretical exponential decay
│   │   │   ├── empirical_vs_theoretical_variance.png # Monte Carlo variance converging to sigma^2 / (2*theta)
│   │   │   ├── estimation_bias_vs_theta.png    # Recovery bias plotted across true mean-reversion speeds
│   │   │   ├── estimation_bias_vs_sigma.png    # Volatility recovery bias plotted across true sigma
│   │   │   └── parameter_recovery_comparison.png # True vs estimated parity boxplots & Hurwicz bias decay
│   │   ├── tables/
│   │   │   ├── discretization_comparison.csv   # Pathwise MAE, RMSE, Max error across time-step sizes
│   │   │   ├── euler_stability_analysis.csv    # Stability regimes, amplification factors, critical dt
│   │   │   ├── moment_validation_results.csv   # Theoretical vs empirical mean/variance at time checkpoints
│   │   │   ├── parameter_recovery_results.csv  # Parameter recovery bias/MAE/RMSE across theta and sigma grid
│   │   │   └── parameter_recovery_by_sample_size.csv # Hurwicz bias decay across sample sizes N
│   │   ├── phase_4_summary.md                  # Comprehensive empirical and theoretical methodology report
│   │   └── phase_4_validation_report.md        # 10-point independent validation audit report (10/10 passed)
│   ├── scripts/
│   │   ├── __init__.py                         # Python package marker
│   │   ├── ou_simulation.py                    # Exact & Euler simulation, coupled engine, stability, moments
│   │   ├── run_phase4.py                       # Pipeline runner for Phase 4
│   │   └── validate_phase4.py                  # Independent Phase 4 verification script
│   ├── tests/
│   │   └── test_phase4.py                      # Unit tests for Phase 4 simulation engine
│   └── run_phase4.py                           # Top-level launcher for Phase 4
├── Phase_5_Trading_Strategy/                   # Phase 5: Trading strategy and out-of-sample backtesting
│   ├── results/
│   │   ├── figures/
│   │   │   ├── equity_curves.png               # Cumulative net equity curves vs. initial capital
│   │   │   ├── drawdown_curves.png             # Portfolio underwater drawdown profiles (%)
│   │   │   ├── spread_signals_HDFCBANK_KOTAKBANK.png # Primary pair spread, thresholds, and trade markers
│   │   │   ├── spread_signals_all_pairs.png    # 3-panel spread and trade visualization
│   │   │   └── performance_comparison.png      # Bar chart comparing Net Return, Sharpe, Drawdown, Win Rate
│   │   ├── tables/
│   │   │   ├── trade_log.csv                   # Ledger of all 18 executed trades (prices, Z-scores, PnL, fees)
│   │   │   ├── daily_portfolio_returns.csv     # Daily portfolio equity, returns, and costs (680 days)
│   │   │   ├── performance_metrics.csv         # Standardized risk-adjusted performance table
│   │   │   └── pair_equity_curves.csv          # Daily equity and drawdown series for each pair
│   │   ├── phase_5_summary.md                  # Comprehensive strategy summary and empirical analysis
│   │   └── phase_5_validation_report.md        # 10-point independent validation audit report (10/10 passed)
│   ├── scripts/
│   │   ├── __init__.py                         # Python package marker
│   │   ├── backtest_engine.py                  # Strategy state machine, spread return normalization, cost accounting
│   │   ├── plot_results.py                     # High-resolution publication-quality plotting utilities
│   │   ├── run_phase5.py                       # Pipeline runner for Phase 5
│   │   └── validate_phase5.py                  # Independent Phase 5 verification script
│   └── tests/
│       └── test_phase5.py                      # Unit tests for Phase 5 backtesting engine
├── Phase_6_Performance_Benchmark/              # Phase 6: Performance metrics & benchmark comparison
│   ├── results/
│   │   ├── figures/
│   │   │   ├── benchmark_equity_curves.png     # Out-of-sample cumulative net equity across strategies
│   │   │   ├── benchmark_drawdowns.png         # Underwater drawdown comparison profiles (%)
│   │   │   ├── fpt_theoretical_surface.png     # Continuous-time OU FPT return rate & trade duration surface
│   │   │   ├── threshold_sensitivity_heatmap.png # 2D parameter heatmaps (Return, Sharpe, Win Rate, Drawdown)
│   │   │   ├── transaction_cost_impact.png     # Strategy profitability & Sharpe decay across cost regimes
│   │   │   ├── stop_loss_sensitivity.png       # Capital protection vs premature stop-out analysis
│   │   │   └── performance_comparison_bar.png  # Multi-metric comparative bar chart
│   │   ├── tables/
│   │   │   ├── performance_comparison.csv      # Complete risk-adjusted performance table across strategies
│   │   │   ├── analytical_optimal_thresholds.csv # Exact continuous-time OU FPT threshold solutions
│   │   │   ├── benchmark_trade_log.csv         # Consolidated execution records for all benchmark strategies
│   │   │   ├── benchmark_equity_curves.csv     # Daily comparative net equity series
│   │   │   ├── threshold_sensitivity_grid.csv  # 2D parameter matrix across entry and exit thresholds
│   │   │   ├── transaction_cost_sensitivity.csv# Performance decay across [0, 30] bps per-leg costs
│   │   │   └── stop_loss_sensitivity.csv       # Tail risk metrics across stop barrier levels
│   │   ├── phase_6_summary.md                  # Comprehensive academic methodology & performance report
│   │   └── phase_6_validation_report.md        # 10-point independent validation audit report (10/10 passed)
│   ├── scripts/
│   │   ├── __init__.py                         # Python package marker
│   │   ├── optimal_thresholds.py               # Dynkin's ODE solver & exact erfi closed-form FPT cycle engine
│   │   ├── performance_analytics.py            # Institutional metrics suite (Sharpe, Sortino, Calmar, MaxDD, Turnover)
│   │   ├── benchmark_comparison.py             # Multi-strategy out-of-sample backtest engine
│   │   ├── sensitivity_analysis.py             # 2D threshold grid, cost friction frontier, and stop-loss sweeps
│   │   ├── plot_phase6.py                      # Publication-grade plotting utilities (7 figures)
│   │   ├── run_phase6.py                       # Pipeline runner for Phase 6
│   │   └── validate_phase6.py                  # Independent Phase 6 verification audit script
│   ├── tests/
│   │   └── test_phase6.py                      # Unit tests for Phase 6 FPT solver and performance calculators
│   └── run_phase6.py                           # Top-level launcher for Phase 6
├── Implementation_Plan_OU_Pairs_Trading.md     # Master project blueprint and methodology plan
├── requirements.txt                            # Pinned Python package dependencies
└── README.md                                   # Project documentation and current status
```

---

## 5. Summary of Implemented Empirical Results

### 5.1 Phase 1: Data Collection & Quality
- **Stock Universe (6 Tickers):** `HDFCBANK.NS`, `ICICIBANK.NS`, `AXISBANK.NS`, `KOTAKBANK.NS`, `SBIN.NS`, `INDUSINDBK.NS`.
- **Temporal Coverage:** 1 January 2019 to 30 September 2026 (1,915 common trading days).
- **In-Sample Training Window:** 1 January 2019 to 31 December 2023 (**1,235 trading days**).
- **Out-of-Sample Testing Window:** 1 January 2024 to 30 September 2026 (**680 trading days**).
- **Data Integrity:** **0 missing values**, **0 duplicate dates**, **0 non-positive prices**. No forward-fill or interpolation was required.

### 5.2 Phase 2: Pair Selection & Cointegration
All $\binom{6}{2} = 15$ pairs were screened strictly on in-sample training data:

| Stock A | Stock B | Pearson $r$ | Hedge Ratio $\beta$ | Intercept $\alpha$ | $R^2$ | Engle-Granger $p$-value | Residual ADF $p$-value | Mean-Reversion $\theta$ (yr⁻¹) | Half-Life $t_{1/2}$ (days) | Cointegrated ($p < 0.05$)? | Final Selected? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDFCBANK.NS** | **KOTAKBANK.NS** | **0.9076** | **1.0931** | **0.1103** | **0.8237** | **0.0481** | **0.0128** | **5.64** | **31.0** | **Yes** | **Yes (Primary)** |
| KOTAKBANK.NS | SBIN.NS | 0.8143 | 0.2887 | 5.8690 | 0.6631 | 0.0597 | 0.0164 | 4.56 | 38.2 | No | No (Borderline) |
| HDFCBANK.NS | SBIN.NS | 0.8407 | 0.3943 | 5.0970 | 0.7067 | 0.0973 | 0.0300 | 4.00 | 43.6 | No | No (Borderline) |
| ICICIBANK.NS | KOTAKBANK.NS | 0.8988 | 2.1154 | -9.0069 | 0.8078 | 0.2010 | 0.0719 | 3.32 | 52.5 | No | No |
| ICICIBANK.NS | SBIN.NS | 0.9484 | 0.8874 | 1.4880 | 0.8994 | 0.2024 | 0.0749 | 2.99 | 58.3 | No | No |
| HDFCBANK.NS | ICICIBANK.NS | 0.9293 | 0.4359 | 4.4984 | 0.8637 | 0.2026 | 0.0726 | 3.01 | 58.0 | No | No |
| AXISBANK.NS | KOTAKBANK.NS | 0.8351 | 1.1554 | -1.8679 | 0.6974 | 0.2520 | 0.0946 | 2.92 | 59.7 | No | No |
| AXISBANK.NS | SBIN.NS | 0.8906 | 0.4912 | 3.8643 | 0.7932 | 0.3475 | 0.1515 | 2.42 | 72.0 | No | No |
| AXISBANK.NS | ICICIBANK.NS | 0.9388 | 0.5484 | 3.2081 | 0.8814 | 0.4855 | 0.2256 | 1.72 | 101.3 | No | No |
| HDFCBANK.NS | INDUSINDBK.NS | 0.4900 | 0.2319 | 5.7533 | 0.2401 | 0.5593 | 0.2864 | 1.40 | 124.8 | No | No |
| HDFCBANK.NS | AXISBANK.NS | 0.8937 | 0.6946 | 2.8251 | 0.7986 | 0.5753 | 0.2969 | 1.32 | 132.7 | No | No |
| KOTAKBANK.NS | INDUSINDBK.NS | 0.3709 | 0.1705 | 6.3533 | 0.1376 | 0.7301 | 0.4441 | 0.97 | 180.3 | No | No |
| ICICIBANK.NS | INDUSINDBK.NS | 0.5369 | 0.5126 | 3.2205 | 0.2882 | 0.7593 | 0.4790 | 0.93 | 188.4 | No | No |
| AXISBANK.NS | INDUSINDBK.NS | 0.6033 | 0.5985 | 2.6860 | 0.3640 | 0.7659 | 0.4919 | 1.02 | 171.6 | No | No |
| SBIN.NS | INDUSINDBK.NS | 0.5323 | 0.5714 | 2.0152 | 0.2833 | 0.9022 | 0.6690 | 0.69 | 255.0 | No | No |

**Key Finding:** Correlation $\ne$ Cointegration. While `ICICIBANK.NS - SBIN.NS` achieves the highest correlation ($r = 0.9484$), it completely fails cointegration testing ($p = 0.2024$). `HDFCBANK.NS - KOTAKBANK.NS` displays genuine cointegration ($p = 0.0481 < 0.05$; residual ADF $p = 0.0128$), an estimated hedge ratio $\beta = 1.0931$, and an in-sample half-life of $31.0$ trading days.

### 5.3 Phase 3: Rolling Ornstein–Uhlenbeck Parameter Estimation
Using a fixed hedge ratio $\beta$ and rolling windows of exactly 120 trading days ($\Delta t = 1/252$ yr) across all $1,796$ windows (1,116 train, 680 test):

| Pair | Fixed $\beta$ | Total Windows | Valid Windows | Invalid Windows | % Valid Total | % Valid Train | % Valid Test | Mean $\theta$ (yr⁻¹) | Median $\theta$ (yr⁻¹) | Mean $\mu$ | Mean $\sigma$ | Mean $\sigma_{eq}$ | Mean Half-Life (days) | Median Half-Life (days) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDFCBANK.NS - KOTAKBANK.NS** | **1.0931** | **1,796** | **1,792** | **4** | **99.78%** | **99.82%** | **99.71%** | **16.25** | **13.52** | **0.1497** | **0.2513** | **0.0503** | **16.6** | **12.9** |
| HDFCBANK.NS - SBIN.NS | 0.3943 | 1,796 | 1,725 | 71 | 96.05% | 99.46% | 90.44% | 14.05 | 12.86 | 4.1028 | 0.2143 | 0.0491 | 27.3 | 13.6 |
| KOTAKBANK.NS - SBIN.NS | 0.2887 | 1,796 | 1,759 | 37 | 97.94% | 98.84% | 96.47% | 14.56 | 12.25 | 4.1040 | 0.2411 | 0.0544 | 25.8 | 14.3 |

**Key Insights for Primary Pair (`HDFCBANK.NS - KOTAKBANK.NS`):**
- **Validity Rate:** **99.78% of windows** satisfy the mean-reversion criterion $0 < b < 1$. Only 4 windows (0.22%) displayed local trending pauses ($b \ge 1.0$), which were explicitly flagged as invalid with `NaN` parameters (zero silent clipping).
- **Mean-Reversion Speed ($\theta$):** Averages $16.25 \text{ yr}^{-1}$ (median $13.52 \text{ yr}^{-1}$), confirming active mean-reverting drift.
- **Half-Life Distribution ($t_{1/2}$):** Median half-life of **$12.9 \text{ trading days}$** (mean $16.6$ days, IQR $[9.3, 19.8]$ days), confirming an optimal holding horizon for statistical arbitrage.
- **Signal Frequency:** On the standardized spread $s_t$, **38 trading days** triggered long entries ($s_t < -2.0$), **44 trading days** triggered short entries ($s_t > +2.0$), and only **5 trading days** breached the structural stop-loss boundary ($|s_t| \ge 3.5$).

### 5.4 Phase 4: OU Process Simulation and Numerical Validation
1. **Coupled Pathwise Error Scaling:** Across $500$ coupled paths ($\theta = 13.52, \sigma = 0.25$), pathwise error between exact and Euler discretisation scales with time step:
   - Daily ($\Delta t = 1/252$): $\text{MAE} = \mathbf{0.001010}$, $\text{RMSE} = \mathbf{0.001309}$ (Monotonically Stable, $g = +0.9463$).
   - Weekly ($\Delta t = 5/252$): $\text{MAE} = \mathbf{0.005427}$, $\text{RMSE} = \mathbf{0.007067}$.
   - Monthly ($\Delta t = 21/252$): $\text{MAE} = \mathbf{0.033729}$, $\text{RMSE} = \mathbf{0.046645}$ (Oscillatory Stable, $g = -0.1267$).
   - Quarterly ($\Delta t = 63/252$): $\text{MAE} = \mathbf{2.678457}$, $\text{RMSE} = \mathbf{4.038285}$ (**Linearly Unstable / Explosive**, $g = -2.3800$).
2. **Euler–Maruyama Stability Threshold:** Verified linear stability threshold $\Delta t < \frac{2}{\theta}$. When $\theta \Delta t \ge 2$, Euler–Maruyama suffers catastrophic numerical overflow, whereas exact discretisation remains unconditionally stable.
3. **Monte Carlo Theoretical Moment Convergence ($M = 10,000$ Paths):**
   - Empirical expectation tracks theoretical decay $\mathbb{E}[X_t] = \mu + (X_0 - \mu)e^{-\theta t}$ within $< 0.15\%$ relative error across all time points.
   - Empirical asymptotic variance at $t = 3.0$ yr reaches $\mathbf{0.002293}$, closely matching the theoretical bound $\sigma_{eq}^2 = \frac{\sigma^2}{2\theta} = \frac{0.25^2}{2 \times 13.52} = \mathbf{0.002311}$ (relative difference $0.77\%$).
4. **Parameter Recovery & Hurwicz Bias Analysis:**
   - In short windows ($N = 120$), OLS AR(1) estimation exhibits positive Hurwicz bias in $\widehat{\theta}$ ($\mathbb{E}[\widehat{\theta}] = 20.07$ vs. true $10.0$).
   - As sample size increases ($N = 252 \to 500 \to 1235$), the bias decays smoothly to $+0.70$ ($+7\%$).
   - Volatility $\sigma$ is estimated with near-zero bias across all sample sizes ($\widehat{\sigma} = 0.251$ vs. $0.250$).

### 5.5 Phase 5: Trading Strategy and Out-of-Sample Backtesting
The mean-reversion trading strategy was executed on the out-of-sample period (1 January 2024 to 30 September 2026, 680 trading days) under strict zero look-ahead bias, fixed training hedge ratios $\beta$, and 10 bps per-leg transaction costs (20 bps entry, 20 bps exit = 40 bps round-trip):

| Pair / Portfolio | Total Trades | Win Rate | Avg Hold (Days) | Cum. Gross Return | Cum. Net Return | Ann. Net Return | Ann. Volatility | Sharpe ($r_f=0$) | Sharpe ($r_f=6.5\%$) | Max Drawdown (Net) | Total Costs | Ann. Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HDFCBANK.NS - KOTAKBANK.NS`** | 3 | 0.0% | 25.7 | -2.83% | **-3.99%** | -1.50% | 2.77% | -0.54 | -2.88 | **4.76%** | 1.20% | 2.22x |
| **`HDFCBANK.NS - SBIN.NS`** | 9 | **88.9%** | 8.8 | +17.15% | **+13.02%** | +4.64% | 5.05% | **+0.92** | -0.37 | 7.03% | 3.60% | 6.67x |
| **`KOTAKBANK.NS - SBIN.NS`** | 6 | 33.3% | 13.5 | -7.73% | **-9.94%** | -3.81% | 5.20% | -0.73 | -1.98 | 10.03% | 2.40% | 4.45x |
| **`COMBINED PORTFOLIO`** (Equal Weight 1/3) | **18** | **55.6%** | **13.2** | **+1.84%** | **-0.58%** | **-0.21%** | **2.45%** | **-0.09** | **-2.74** | **3.58%** | **7.20%** | **13.34x** |

**Key Empirical Findings & Financial Takeaways:**
1. **Transaction Cost Drag:** Round-trip transaction friction of 40 bps ($0.40\%$) per trade consumed **7.20%** of gross portfolio capital across 18 trades. In Trade 1 of the primary pair, a gross gain of +0.22% resulted in a -0.18% net loss, proving that statistical arbitrage alpha is highly sensitive to broker commissions.
2. **Rolling Parameter Adaptation vs. Physical Reversion:** In extended trades (e.g. Trade 2 held 46 days), the rolling 120-day mean $\hat{\mu}_t$ adapted upward toward the drifting spread $X_t$, triggering a take-profit exit ($|s_t| \le 0.5$) due to parameter movement rather than physical spread convergence.
3. **Stop-Loss Protection in Structural Divergence:** The stop-loss rule ($|s_t| \ge 3.5$) successfully protected capital when PSU banking momentum caused SBI to surge in early 2026. This preserved capital for subsequent mean-reverting swings on `HDFCBANK - SBIN` (+13.02% net return, 88.9% win rate).
4. **Portfolio Diversification Compresses Downside Risk:** Combining pairs in an equal-weighted portfolio compressed maximum net drawdown to just **3.58%** (compared to 7.03% for HDFC-SBI and 10.03% for Kotak-SBI) and reduced annualized volatility to **2.45%**.
5. **Audited Integrity:** 100% of trades satisfied $\text{Entry\_Date} > \text{Signal\_Date}$, with zero look-ahead bias and exact balance-sheet compounding verified in `validate_phase5.py`.

### 5.6 Phase 6: Performance Metrics & Benchmark Comparison
Phase 6 grounds pairs trading threshold selection in the continuous-time OU first-passage-time (FPT) problem (Dynkin's infinitesimal generator ODE and Bertram 2010), comparing the fixed heuristic strategy against analytical optimal thresholds, in-sample calibrated thresholds, and passive market benchmarks:

| Strategy | Target Pair / Portfolio | Total Trades | Win Rate | Avg Hold (Days) | Cum. Net Return | Ann. Net Return | Ann. Volatility | Sharpe ($r_f=0$) | Sharpe ($r_f=6.5\%$) | Max Drawdown | Turnover |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Benchmark (2.0 / 0.5)** | `HDFCBANK - KOTAKBANK` | 3 | 0.0% | 25.7 | -3.99% | -1.50% | 2.77% | -0.54 | -2.88 | 4.76% | 2.22x |
| **Analytical Optimal (FPT)** | `HDFCBANK - KOTAKBANK` | 1 | **100.0%** | 47.0 | **+11.87%** | **+4.24%** | 11.94% | **+0.36** | -0.19 | 13.85% | 0.74x |
| **In-Sample Calibrated (1.75 / 0.25)** | `HDFCBANK - KOTAKBANK` | 4 | 50.0% | 20.2 | **+1.02%** | **+0.37%** | 4.98% | **+0.08** | -1.23 | 5.98% | 2.96x |
| **Fixed Benchmark (2.0 / 0.5)** | `HDFCBANK - SBIN` | 9 | **88.9%** | 8.8 | **+13.02%** | **+4.64%** | 5.05% | **+0.92** | -0.37 | 7.03% | 6.67x |
| **Fixed Benchmark (2.0 / 0.5)** | `COMBINED PORTFOLIO` | **18** | **55.6%** | **13.2** | **-0.58%** | **-0.21%** | **2.45%** | **-0.09** | **-2.74** | **3.58%** | **13.34x** |
| **Buy & Hold HDFC Bank** | Market Benchmark | - | - | - | -12.83% | -4.97% | 20.76% | -0.24 | -0.55 | 31.04% | 0.00x |
| **Buy & Hold Kotak Bank** | Market Benchmark | - | - | - | +9.65% | +3.48% | 23.01% | +0.15 | -0.13 | 22.03% | 0.00x |
| **Equal-Weighted Banking Basket**| Market Benchmark | - | - | - | +9.63% | +3.47% | 16.96% | +0.20 | -0.18 | 17.95% | 0.00x |
| **Cash Risk-Free Asset** | Cash Benchmark (6.5% p.a.) | - | - | - | +18.49% | +6.50% | 0.02% | +426.57 | 0.00 | 0.00% | 0.00x |

**Core Analytical and Empirical Takeaways:**
1. **Dynkin's ODE Theoretical Holding Time Precision:** Solving the continuous-time first-passage-time ODE $\mathbb{E}[T_{\text{trade}}] = \frac{\sqrt{2\pi}}{\theta} \int_{s_x}^{s_e} e^{u^2/2}(1 - \Phi(u))du$ under heuristic thresholds yields an expected holding time of **13.99 trading days**, tracking the empirical out-of-sample portfolio holding average of **13.2 trading days** with extreme fidelity.
2. **Analytical FPT Outperformance on Cointegrated Pair:** On the primary cointegrated pair (`HDFCBANK - KOTAKBANK`), the Analytical Optimal FPT strategy turned a -3.99% net loss under heuristic thresholds into a **+11.87% net gain** (Sharpe +0.36) by entering earlier ($s_e^* = 1.25$) and capturing the full mean-reverting swing back to equilibrium ($s_x^* = 0.0$).
3. **Transaction Cost Friction Frontier:** The strategy generates positive risk-adjusted returns for commissions up to **12.5 bps per leg** (50 bps round-trip). Beyond 15 bps per leg, transaction costs consume more than 100% of statistical arbitrage alpha.
4. **2D Threshold Grid Dynamics:** Exit thresholds close to the equilibrium mean ($s_x \le 0.25$) consistently maximize cumulative profit, while entry thresholds below $1.5\sigma_{eq}$ suffer from excessive turnover and commission drag.
5. **Audited Verification:** All 10 independent verification checks passed (10/10, 100%), confirming zero look-ahead bias, balance-sheet compounding consistency, and exact closed-form `erfi` cycle time validation.

---

## 6. Technologies & Dependencies

The implementation relies on standard scientific Python libraries:

| Library | Version Requirement | Purpose in Project |
| :--- | :--- | :--- |
| **Python** | `>= 3.10` | Base runtime environment |
| **NumPy** | `>= 2.0.0` | Vectorized numerical linear algebra & mathematical operations |
| **Pandas** | `>= 2.0.0` | Time-series alignment, data frames, and CSV input/output |
| **SciPy** | `>= 1.14.0` | Scientific computing, optimization, and statistical distributions |
| **Statsmodels**| `>= 0.14.0` | OLS regressions, Augmented Dickey-Fuller & Engle–Granger tests |
| **yfinance** | `>= 0.2.40` | Market data acquisition for NSE stock tickers |
| **Matplotlib** | `>= 3.9.0` | Time-series charting, residual spread visualization, figures |
| **Seaborn** | `>= 0.13.0` | Correlation heatmaps and styled statistical graphics |
| **tabulate** | `>= 0.9.0` | Terminal and markdown table formatting |

---

## 7. Installation & Execution Instructions

### 7.1 Environment Setup
```bash
# 1. Clone the repository and navigate to the root directory:
cd ou-mean-reversion-statistical-arbitrage

# 2. Create and activate a Python virtual environment:
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies:
pip install -r requirements.txt
```

### 7.2 Running Phase 1 (Data Preprocessing)
```bash
# Execute Phase 1 pipeline (uses cached raw data):
python Phase_1_Data_Preprocessing/run_phase1.py

# Force re-download from Yahoo Finance:
python Phase_1_Data_Preprocessing/scripts/run_phase1.py --force-download

# Run Phase 1 independent validation audit:
python Phase_1_Data_Preprocessing/scripts/validate_phase1.py

# Run Phase 1 automated unit tests:
python -m unittest Phase_1_Data_Preprocessing/tests/test_phase1.py
```

### 7.3 Running Phase 2 (Pair Selection & Cointegration)
```bash
# Execute Phase 2 pipeline:
python Phase_2_Pair_Selection/run_phase2.py

# Run Phase 2 independent validation audit:
python Phase_2_Pair_Selection/scripts/validate_phase2.py

# Run Phase 2 automated unit tests:
python -m unittest Phase_2_Pair_Selection/tests/test_phase2.py
```

### 7.4 Running Phase 3 (OU Parameter Estimation)
```bash
# Execute Phase 3 rolling estimation pipeline:
python Phase_3_OU_Estimation/run_phase3.py

# Run Phase 3 independent validation audit:
python Phase_3_OU_Estimation/scripts/validate_phase3.py

# Run Phase 3 automated unit tests:
python -m unittest Phase_3_OU_Estimation/tests/test_phase3.py
```

### 7.5 Running Phase 4 (OU Simulation & Numerical Validation)
```bash
# Execute Phase 4 full simulation pipeline (all 3 experiments):
python Phase_4_OU_Simulation/run_phase4.py

# Run Phase 4 independent validation audit:
python Phase_4_OU_Simulation/scripts/validate_phase4.py

# Run Phase 4 automated unit tests:
python -m unittest Phase_4_OU_Simulation/tests/test_phase4.py
```

### 7.6 Running Phase 5 (Trading Strategy & Out-of-Sample Backtesting)
```bash
# Execute Phase 5 backtesting pipeline across all pairs and combined portfolio:
python Phase_5_Trading_Strategy/scripts/run_phase5.py

# Run Phase 5 independent validation audit (10/10 checks):
python Phase_5_Trading_Strategy/scripts/validate_phase5.py

# Run Phase 5 automated unit tests:
python -m unittest Phase_5_Trading_Strategy/tests/test_phase5.py
```

### 7.7 Running Phase 6 (Performance Metrics & Benchmark Comparison)
```bash
# Execute Phase 6 complete pipeline (FPT optimization, benchmark backtests, sensitivity sweeps):
python Phase_6_Performance_Benchmark/run_phase6.py

# Run Phase 6 independent validation audit (10/10 checks):
python Phase_6_Performance_Benchmark/scripts/validate_phase6.py

# Run Phase 6 automated unit tests:
python -m unittest Phase_6_Performance_Benchmark/tests/test_phase6.py
```

---

## 8. Verified Output Artifacts & Locations

| Phase | Output Type | File Path | Description |
| :---: | :--- | :--- | :--- |
| **Phase 1** | Processed Data | `Phase_1_Data_Preprocessing/data/processed/cleaned_prices.csv` | 1,915 daily aligned adjusted closing prices |
| **Phase 1** | Processed Data | `Phase_1_Data_Preprocessing/data/processed/log_prices.csv` | Natural logarithm price series |
| **Phase 1** | Processed Data | `Phase_1_Data_Preprocessing/data/processed/train_log_prices.csv` | In-sample training dataset (1,235 obs) |
| **Phase 1** | Processed Data | `Phase_1_Data_Preprocessing/data/processed/test_log_prices.csv` | Out-of-sample testing dataset (680 obs) |
| **Phase 1** | Table | `Phase_1_Data_Preprocessing/results/tables/data_quality_summary.csv` | Cleanliness metrics and descriptive statistics |
| **Phase 1** | Figure | `Phase_1_Data_Preprocessing/results/figures/asset_prices.png` | Adjusted price series with train/test division |
| **Phase 1** | Report | `Phase_1_Data_Preprocessing/results/phase_1_validation_report.md` | 10-point audit verifying zero data defects |
| **Phase 2** | Table | `Phase_2_Pair_Selection/results/tables/all_pairs_analysis.csv` | Full statistical evaluation of all 15 candidate pairs |
| **Phase 2** | Table | `Phase_2_Pair_Selection/results/tables/cointegrated_pairs.csv` | Cointegrated pairs meeting $p < 0.05$ threshold |
| **Phase 2** | Table | `Phase_2_Pair_Selection/results/tables/final_selected_pairs.csv` | Final selected pairs meeting cointegration & half-life |
| **Phase 2** | Figure | `Phase_2_Pair_Selection/results/figures/correlation_matrix.png` | 6x6 Pearson log-price correlation matrix heatmap |
| **Phase 2** | Figure | `Phase_2_Pair_Selection/results/figures/correlation_vs_cointegration.png` | Scatter plot: Correlation vs. Engle-Granger $p$-value |
| **Phase 2** | Figure | `Phase_2_Pair_Selection/results/figures/cointegration_pvalues.png` | Ranked bar chart of $p$-values vs. 5% significance |
| **Phase 2** | Figure | `Phase_2_Pair_Selection/results/figures/selected_pair_spreads.png` | In-sample synthetic spread trajectory for selected pair |
| **Phase 2** | Report | `Phase_2_Pair_Selection/results/phase_2_summary.md` | Theoretical and empirical methodology summary |
| **Phase 2** | Report | `Phase_2_Pair_Selection/results/phase_2_validation_report.md` | 7-point independent validation audit (0 discrepancies) |
| **Phase 3** | Table | `Phase_3_OU_Estimation/results/tables/rolling_params_HDFCBANK_KOTAKBANK.csv` | Primary pair rolling parameter estimates across all 1,796 windows |
| **Phase 3** | Table | `Phase_3_OU_Estimation/results/tables/combined_rolling_parameters.csv` | Combined rolling parameter estimates for all selected pairs |
| **Phase 3** | Table | `Phase_3_OU_Estimation/results/tables/standardised_spreads.csv` | Contemporaneous ($s_t$) and lagged ($s_t^{\text{lagged}}$) Z-scores |
| **Phase 3** | Table | `Phase_3_OU_Estimation/results/tables/rolling_parameter_summary.csv` | Statistical summary, distribution moments & validity percentages |
| **Phase 3** | Figure | `Phase_3_OU_Estimation/results/figures/parameter_dashboard_HDFCBANK_KOTAKBANK.png` | 6-panel comprehensive diagnostic dashboard for primary pair |
| **Phase 3** | Figure | `Phase_3_OU_Estimation/results/figures/standardised_spread_HDFCBANK_KOTAKBANK.png` | Standardized spread with $0, \pm 2, \pm 3.5$ trading trigger lines |
| **Phase 3** | Figure | `Phase_3_OU_Estimation/results/figures/rolling_halflife_HDFCBANK_KOTAKBANK.png` | Rolling half-life with $[5, 60]$ trading day acceptance boundaries |
| **Phase 3** | Report | `Phase_3_OU_Estimation/results/phase_3_summary.md` | Theoretical formulation, exact discretisation, and findings report |
| **Phase 3** | Report | `Phase_3_OU_Estimation/results/phase_3_validation_report.md` | 10-point independent validation audit report (10/10 passed) |
| **Phase 4** | Table | `Phase_4_OU_Simulation/results/tables/discretization_comparison.csv` | Pathwise MAE, RMSE, Max error across time-step sizes |
| **Phase 4** | Table | `Phase_4_OU_Simulation/results/tables/euler_stability_analysis.csv` | Stability regimes, amplification factors, critical dt |
| **Phase 4** | Table | `Phase_4_OU_Simulation/results/tables/moment_validation_results.csv` | Theoretical vs empirical mean/variance across time checkpoints |
| **Phase 4** | Table | `Phase_4_OU_Simulation/results/tables/parameter_recovery_results.csv` | Parameter recovery bias/MAE/RMSE across theta and sigma grid |
| **Phase 4** | Table | `Phase_4_OU_Simulation/results/tables/parameter_recovery_by_sample_size.csv` | Hurwicz bias decay across sample sizes N |
| **Phase 4** | Figure | `Phase_4_OU_Simulation/results/figures/exact_vs_euler_paths.png` | 4-panel coupled comparison (Daily, Monthly, Quarterly, Unstable) |
| **Phase 4** | Figure | `Phase_4_OU_Simulation/results/figures/simulation_error_vs_timestep.png` | Numerical MAE and RMSE scaling with time-step size dt |
| **Phase 4** | Figure | `Phase_4_OU_Simulation/results/figures/empirical_vs_theoretical_mean.png` | 10,000-path Monte Carlo mean tracking theoretical exponential decay |
| **Phase 4** | Figure | `Phase_4_OU_Simulation/results/figures/empirical_vs_theoretical_variance.png` | Monte Carlo variance converging to stationary sigma^2 / (2*theta) |
| **Phase 4** | Figure | `Phase_4_OU_Simulation/results/figures/parameter_recovery_comparison.png` | Parity boxplots and Hurwicz bias decay across sample sizes |
| **Phase 4** | Report | `Phase_4_OU_Simulation/results/phase_4_summary.md` | Comprehensive simulation and numerical validation report |
| **Phase 4** | Report | `Phase_4_OU_Simulation/results/phase_4_validation_report.md` | 10-point independent validation audit report (10/10 passed) |
| **Phase 5** | Table | `Phase_5_Trading_Strategy/results/tables/trade_log.csv` | Full transaction ledger for all 18 executed trades |
| **Phase 5** | Table | `Phase_5_Trading_Strategy/results/tables/daily_portfolio_returns.csv` | 680-day daily portfolio net/gross returns, costs, and equity |
| **Phase 5** | Table | `Phase_5_Trading_Strategy/results/tables/performance_metrics.csv` | Sharpe, MaxDD, Win Rate, Turnover metrics across pairs |
| **Phase 5** | Table | `Phase_5_Trading_Strategy/results/tables/pair_equity_curves.csv` | Daily net/gross equity and drawdown series per pair |
| **Phase 5** | Figure | `Phase_5_Trading_Strategy/results/figures/equity_curves.png` | Net cumulative equity curves vs. initial capital |
| **Phase 5** | Figure | `Phase_5_Trading_Strategy/results/figures/drawdown_curves.png` | Underwater drawdown profiles across pairs and portfolio |
| **Phase 5** | Figure | `Phase_5_Trading_Strategy/results/figures/spread_signals_HDFCBANK_KOTAKBANK.png` | Primary pair spread with threshold bands and trade markers |
| **Phase 5** | Figure | `Phase_5_Trading_Strategy/results/figures/spread_signals_all_pairs.png` | 3-panel spread and trade visualization across all pairs |
| **Phase 5** | Figure | `Phase_5_Trading_Strategy/results/figures/performance_comparison.png` | Bar chart comparing Return, Sharpe, Drawdown, Win Rate |
| **Phase 5** | Report | `Phase_5_Trading_Strategy/results/phase_5_summary.md` | Strategy methodology and out-of-sample empirical analysis |
| **Phase 5** | Report | `Phase_5_Trading_Strategy/results/phase_5_validation_report.md` | 10-point independent validation audit report (10/10 passed) |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/performance_comparison.csv` | Standardized risk-adjusted performance table across strategies |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/analytical_optimal_thresholds.csv` | Exact continuous-time OU FPT threshold solutions & theoretical moments |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/benchmark_trade_log.csv` | Consolidated execution trade records across all benchmark strategies |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/benchmark_equity_curves.csv` | Daily comparative net equity series for strategies and assets |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/threshold_sensitivity_grid.csv` | 2D parameter matrix across entry and exit threshold combinations |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/transaction_cost_sensitivity.csv` | Net return and Sharpe decay across [0, 30] bps per-leg transaction costs |
| **Phase 6** | Table | `Phase_6_Performance_Benchmark/results/tables/stop_loss_sensitivity.csv` | Tail risk metrics and win rates across stop barrier levels |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/benchmark_equity_curves.png` | Out-of-sample cumulative net equity across strategies and benchmarks |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/benchmark_drawdowns.png` | Underwater drawdown comparison profiles (%) |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/fpt_theoretical_surface.png` | Continuous-time OU FPT return rate & trade duration surface |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/threshold_sensitivity_heatmap.png` | 2D parameter heatmaps (Return, Sharpe, Win Rate, Drawdown) |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/transaction_cost_impact.png` | Strategy profitability & Sharpe decay across cost regimes |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/stop_loss_sensitivity.png` | Capital protection vs premature stop-out analysis |
| **Phase 6** | Figure | `Phase_6_Performance_Benchmark/results/figures/performance_comparison_bar.png` | Multi-metric comparative bar chart across strategies |
| **Phase 6** | Report | `Phase_6_Performance_Benchmark/results/phase_6_summary.md` | Comprehensive academic methodology & performance report |
| **Phase 6** | Report | `Phase_6_Performance_Benchmark/results/phase_6_validation_report.md` | 10-point independent validation audit report (10/10 passed) |

---

## 9. Planned Future Phases

The remaining research phases outlined in `Implementation_Plan_OU_Pairs_Trading.md` will be sequentially developed:

1. **Phase 7: Project Report & Academic Presentation (Upcoming):**
   - Produce the final 3–4 page academic project report and viva presentation slides for MAC F244 defense.

