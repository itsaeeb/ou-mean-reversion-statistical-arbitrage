# Implementation Plan
## Stochastic Modelling of Statistical Arbitrage Using Ornstein–Uhlenbeck Processes: A Mean-Reversion Based Pairs Trading Strategy

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Final submission deadline:** 15 November 2026  
**Implementation language:** Python  
**Development environment:** Antigravity IDE

---

## 1. Project objective

Implement and evaluate a pairs-trading strategy in which the log-price spread of two cointegrated financial assets is modelled using an Ornstein–Uhlenbeck (OU) process.

The project must demonstrate the connection between stochastic calculus and computational finance. It should include:

1. Mathematical formulation of the OU process and its theoretical properties.
2. Pair selection using correlation, OLS regression and the Engle–Granger cointegration approach.
3. Estimation of OU parameters from historical data.
4. Validation of parameter estimation using simulated OU paths with known parameters.
5. Comparison of exact OU discretisation and Euler–Maruyama simulation.
6. A mean-reversion-based trading strategy and out-of-sample backtest.
7. Sensitivity and robustness experiments.
8. Interpretation of results, limitations and conclusions.

The implementation must do more than reproduce formulas. Numerical experiments should investigate how changes in model parameters, assets, thresholds, estimation windows and transaction costs affect the results.

---

## 2. Submission requirements

The final submission must contain:

### 2.1 Source code
- Well-structured, readable and properly commented code.
- Python is the selected implementation language.
- Code should be organised into separate modules.
- Include instructions for installing dependencies and running the project.

### 2.2 Project report
- Length: 3–4 pages.
- Describe the mathematical model, numerical methodology, results and conclusions.
- Reference all mathematical models, algorithms and datasets appropriately.

### 2.3 Presentation / viva
- Duration: 15–20 minutes.
- Explain the mathematical formulation, numerical methodology and key findings.
- Be prepared to justify modelling and implementation choices.

### 2.4 Evaluation scheme

| Criterion | Marks |
|---|---:|
| Report | 5 |
| Mathematical formulation and explanation of model | 5 |
| Numerical results, analysis and interpretation | 5 |
| Viva / Presentation | 5 |
| **Total** | **20** |

---

## 3. Mathematical model

### 3.1 Log-price spread

For two assets A and B with prices \(P_t^A\) and \(P_t^B\), define the log-price spread:

\[
X_t = \ln(P_t^A) - \beta \ln(P_t^B)
\]

The hedge ratio \(\beta\) is estimated by OLS regression on log prices, using the training period only.

The regression is:

\[
\ln(P_t^A) = \alpha + \beta \ln(P_t^B) + \epsilon_t
\]

The spread is then constructed as the regression residual:

\[
X_t = \ln(P_t^A) - \beta \ln(P_t^B)
\]

### 3.2 Ornstein–Uhlenbeck process

Model the spread using:

\[
dX_t = \theta(\mu-X_t)\,dt + \sigma\,dW_t,
\qquad \theta,\sigma>0
\]

where:

- \(\theta\): mean-reversion speed.
- \(\mu\): long-run equilibrium mean.
- \(\sigma\): volatility parameter.
- \(W_t\): standard Brownian motion.

### 3.3 Theoretical properties to implement and explain

**Explicit solution**

\[
X_t = \mu + (X_0-\mu)e^{-\theta t}
+\sigma\int_0^t e^{-\theta(t-s)}\,dW_s
\]

This follows by applying Itô's lemma to \(e^{\theta t}X_t\).

**Expectation**

\[
\mathbb{E}[X_t] = \mu+(X_0-\mu)e^{-\theta t}
\]

**Variance**

\[
\operatorname{Var}(X_t)
=\frac{\sigma^2}{2\theta}(1-e^{-2\theta t})
\]

**Stationary distribution**

\[
X_t \xrightarrow[t\to\infty]{d}
N\left(\mu,\frac{\sigma^2}{2\theta}\right)
\]

**Equilibrium standard deviation**

\[
\sigma_{eq}=\frac{\sigma}{\sqrt{2\theta}}
\]

**Half-life**

\[
t_{1/2}=\frac{\ln 2}{\theta}
\]

The OU parameters are annualised when the time step is measured in years. With \(\Delta=1/252\), convert half-life into trading days by multiplying the half-life in years by 252.

### 3.4 Exact discretisation

For time step \(\Delta\):

\[
X_{t+\Delta}
=\mu(1-e^{-\theta\Delta})
+e^{-\theta\Delta}X_t+\epsilon_t
\]

where:

\[
\epsilon_t\sim N\left(
0,\frac{\sigma^2(1-e^{-2\theta\Delta})}{2\theta}
\right)
\]

This is an AR(1) model. Write it as:

\[
X_{t+\Delta}=a+bX_t+\epsilon_t
\]

Then:

\[
b=e^{-\theta\Delta},
\qquad
a=\mu(1-b)
\]

The corresponding parameter estimates are:

\[
\widehat{\theta}=-\frac{\ln(\widehat b)}{\Delta}
\]

\[
\widehat{\mu}=\frac{\widehat a}{1-\widehat b}
\]

\[
\widehat{\sigma}
=
\sqrt{
\frac{2\widehat{\theta}\,\widehat{\operatorname{Var}}(\epsilon_t)}
{1-\widehat b^2}
}
\]

Use \(\Delta=1/252\) and estimate the AR(1) regression using least squares / the equivalent Gaussian maximum-likelihood approach.

**Numerical safeguards:**
- Check that \(0<\widehat b<1\) before recovering valid mean-reverting OU parameters.
- Handle invalid or insufficient rolling windows explicitly.
- Do not silently accept invalid estimates.
- Document how residual variance is calculated.
- Keep units consistent: \(\theta\) and \(\sigma\) are annualised under this time convention.

---

## 4. Data and experimental design

### 4.1 Assets

Use daily adjusted closing prices for the following Nifty Bank constituents:

- HDFC Bank
- ICICI Bank
- Axis Bank
- Kotak Mahindra Bank
- State Bank of India (SBI)
- IndusInd Bank

Retrieve data in Python using `yfinance` and the appropriate NSE ticker symbols.

### 4.2 Date ranges

| Purpose | Period |
|---|---|
| Training / pair selection / initial model estimation | 1 January 2019 – 31 December 2023 |
| Out-of-sample testing | 1 January 2024 – 30 September 2026 |

Use training data only for pair selection and the initial hedge-ratio estimation. Do not use the test period to choose pairs or tune thresholds.

The proposal reports that the data pipeline was previously tested for 2019–2026, with 1,915 trading days and no missing values. Reuse and verify the existing data work rather than unnecessarily rebuilding it.

### 4.3 Data preparation

Implement the following:

1. Download daily adjusted closing prices.
2. Align asset price series on common trading dates.
3. Check missing values, duplicate dates, invalid prices and data consistency.
4. Calculate log prices.
5. Split the data into training and test periods.
6. Save raw and processed data so that experiments are reproducible.

Do not silently forward-fill or otherwise alter missing observations. Document any cleaning decisions.

**Expected outputs:**
- Raw adjusted-price dataset.
- Cleaned and aligned price dataset.
- Log-price dataset.
- Price and log-price plots.
- Summary statistics and data-quality checks.

---

## 5. Implementation phases

Implement the project in the order below. Complete and test one phase before moving to the next.

### Phase 1 — Data collection and preprocessing

**Objective:** Prepare the historical price data.

Tasks:
1. Download adjusted closing prices for all six assets.
2. Align observations by trading date.
3. Check missing values, duplicates and invalid prices.
4. Calculate log prices.
5. Create training and testing datasets.
6. Save datasets and data-quality summaries.

Expected outputs:
- Cleaned price data.
- Log-price data.
- Asset price plots.
- Log-price comparison plots.
- Data-quality report.

### Phase 2 — Pair selection and cointegration

**Objective:** Identify potentially suitable mean-reverting pairs.

There are:

\[
\binom{6}{2}=15
\]

candidate pairs.

Tasks:
1. Calculate the training-period log-price correlation matrix.
2. Screen candidate pairs by correlation.
3. For shortlisted pairs, regress one asset's log price on the other's using OLS.
4. Estimate the hedge ratio \(\beta\).
5. Construct the residual spread.
6. Apply the Engle–Granger approach by performing an ADF test on the regression residuals.
7. Retain pairs with a significant stationarity result at the 5% level.
8. Estimate preliminary OU parameters and calculate half-life.
9. Retain pairs with half-life between 5 and 60 trading days, as specified in the proposal.

Expected outputs:
- Correlation matrix / heatmap.
- Candidate pair comparison table.
- OLS regression results.
- ADF statistics and p-values.
- Selected pairs.
- Preliminary half-life estimates.

**Important:** Use training data only for this phase. Record the test configuration and significance level.

### Phase 3 — OU parameter estimation

**Objective:** Estimate \(\theta\), \(\mu\) and \(\sigma\) for the selected spread.

Tasks:
1. Implement the exact AR(1) discretisation.
2. Estimate AR(1) coefficients by least squares.
3. Recover OU parameters using the formulas in Section 3.4.
4. Use a rolling estimation window of 120 trading days.
5. Calculate equilibrium standard deviation and standardised spread.
6. Store parameter estimates and validity flags for each window.
7. Plot the rolling estimates and discuss their behaviour.

The standardised spread is:

\[
s_t=\frac{X_t-\mu}{\sigma_{eq}}
\]

Expected outputs:
- Estimated OU parameters.
- Rolling estimates of \(\theta,\mu,\sigma\).
- Half-life estimates.
- Equilibrium standard deviation.
- Standardised spread plots.

**Important distinction:** The proposal specifies estimating \(\beta\) by OLS on the training period and estimating OU parameters with a rolling 120-trading-day window. Preserve this distinction in the initial implementation. Do not silently change to a rolling hedge ratio.

### Phase 4 — OU simulation and estimator validation

**Objective:** Validate the OU model and numerical methods independently of market backtesting.

Implement both simulation methods.

**A. Exact discretisation**

\[
X_{t+\Delta}
=\mu(1-e^{-\theta\Delta})+e^{-\theta\Delta}X_t
+\sigma\sqrt{\frac{1-e^{-2\theta\Delta}}{2\theta}}Z_t
\]

where \(Z_t\sim N(0,1)\).

**B. Euler–Maruyama**

\[
X_{t+\Delta}
=X_t+\theta(\mu-X_t)\Delta+\sigma\sqrt{\Delta}Z_t
\]

Experiments:
1. Simulate multiple OU trajectories.
2. Compare exact discretisation and Euler–Maruyama.
3. Repeat for different time steps.
4. Compare empirical mean with theoretical mean.
5. Compare empirical variance with theoretical variance.
6. Vary \(\theta\) and \(\sigma\) and examine the resulting paths.
7. Generate simulated paths with known parameters and estimate the parameters from them.
8. Measure parameter estimation errors and bias.

Use controlled random seeds for reproducibility. Where comparing two methods, use the same underlying Brownian increments where appropriate so the comparison is meaningful.

Expected outputs:
- OU trajectory plots.
- Exact versus Euler–Maruyama comparison.
- Empirical versus theoretical mean and variance.
- Parameter recovery tables.
- Estimation error / bias analysis.

### Phase 5 — Trading strategy and backtesting

**Objective:** Build and evaluate the mean-reversion-based pairs-trading strategy.

Use:

\[
s_t=\frac{X_t-\mu}{\sigma_{eq}}
\]

Benchmark rules from the proposal:

| Signal condition | Action |
|---|---|
| \(s_t>2\) | Short the spread |
| \(s_t<-2\) | Long the spread |
| Spread returns to the exit region, \(|s_t|<0.5\) | Close the position |
| \(|s_t|\geq 3.5\) | Stop-loss |

Interpretation:
- A high positive spread indicates that A is relatively expensive compared with B: short A and long B.
- A sufficiently negative spread indicates that A is relatively cheap: long A and short B.
- Close the position when the spread returns sufficiently close to equilibrium.

Tasks:
1. Generate signals from the standardised spread.
2. Implement position states and entry/exit logic.
3. Convert the hedge ratio into documented portfolio exposures.
4. Calculate daily portfolio returns.
5. Deduct transaction costs.
6. Apply the stop-loss.
7. Record every trade, including entry/exit dates, direction, prices, costs, holding period and profit/loss.
8. Backtest on the out-of-sample period.

The proposal specifies transaction costs of 10 basis points per leg. Charge costs on each traded leg when positions are opened and closed, including reversals.

**Backtesting safeguards:**
- Avoid look-ahead bias: signals must use only information available at the decision time.
- State whether execution occurs at the same close or the next available price; use a consistent convention.
- Document portfolio capital, position sizing, hedge weights and short-selling assumptions.
- Avoid overlapping or duplicated positions unless explicitly intended.
- Define how invalid rolling parameter estimates affect trading.
- Ensure the stop-loss and exit conditions have an unambiguous priority.
- Do not use test-period results to tune the strategy.

Expected outputs:
- Trade log.
- Equity curve.
- Cumulative returns.
- Drawdown plot.
- Portfolio return series.
- Strategy performance summary.

### Phase 6 — Performance metrics and benchmark comparison

Calculate:

| Metric | Purpose |
|---|---|
| Cumulative return | Overall profitability |
| Annualised Sharpe ratio | Risk-adjusted performance |
| Maximum drawdown | Largest peak-to-trough decline |
| Win rate | Percentage of profitable closed trades |
| Average holding period | Average trade duration |
| Turnover | Extent / frequency of trading |

Use clearly stated conventions for annualisation, risk-free rate, return frequency and turnover.

Compare:
1. Fixed benchmark thresholds: entry 2, exit 0.5.
2. Thresholds selected using the analytical criterion proposed in the project.

For the analytical criterion, investigate the OU first-passage-time problem and expected profit per cycle. A possible numerical approach is to solve the associated OU boundary-value problem and compare expected profit per unit time, accounting for transaction costs.

Do not claim analytically optimal thresholds unless the criterion has actually been derived, implemented and validated. If time is limited, present this as a clearly labelled numerical extension rather than silently substituting arbitrary thresholds.

Expected outputs:
- Performance comparison table.
- Equity curves.
- Drawdown comparison.
- Benchmark versus alternative-threshold comparison.
- Interpretation of risk-adjusted performance and trading activity.

### Phase 7 — Sensitivity and robustness analysis

**Objective:** Determine how model assumptions and market conditions affect strategy performance.

Run the following experiments.

**Experiment 1: Mean-reversion speed**

Vary \(\theta\), holding other parameters fixed. Examine:
- Mean-reversion speed.
- Half-life.
- Simulated spread behaviour.
- Trading frequency.
- Strategy performance.

**Experiment 2: Volatility**

Vary \(\sigma\). Examine:
- Spread fluctuations.
- Standardised signals.
- Entry/exit frequency.
- Risk and profitability.

**Experiment 3: Entry and exit thresholds**

Compare combinations such as:

| Entry | Exit |
|---:|---:|
| 1.5 | 0.25 |
| 2.0 | 0.5 |
| 2.5 | 0.5 |
| 2.5 | 1.0 |
| 3.0 | 0.5 |

Keep the stop-loss at 3.5 for the initial comparison.

**Experiment 4: Estimation window**

Compare:
- 60 trading days.
- 120 trading days.
- 250 trading days.

Examine the trade-off between responsiveness and estimation noise.

**Experiment 5: Transaction costs**

Compare:
- 0 bps.
- 5 bps.
- 10 bps.
- 20 bps.
- 30 bps.

Examine whether the strategy remains viable after transaction costs.

**Experiment 6: Pair choice**

Compare results across all eligible selected pairs. Discuss whether performance is specific to one pair or more broadly consistent.

**Experiment 7: Structural breaks**

Use rolling ADF p-values and changes in estimated half-life to investigate periods when the spread may stop behaving as a mean-reverting process.

Examine periods of market stress and discuss the limitations of assuming a constant OU process.

Expected outputs:
- Sensitivity plots.
- Entry/exit threshold heatmap.
- Transaction-cost impact plot.
- Estimation-window comparison.
- Pair-wise performance comparison.
- Rolling ADF and half-life plots.
- Discussion of robustness and failure conditions.

**Avoid overfitting:** Thresholds and other strategy choices must be selected without using the final test-period performance. Clearly distinguish exploratory analysis from out-of-sample evaluation.

---

## 6. Recommended project structure

Use the following modular folder structure. Inspect the existing project before creating files; adapt the structure if equivalent files already exist. Do not duplicate existing functionality.

```text
OU_Pairs_Trading/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── pair_selection.py
│   ├── ou_model.py
│   ├── parameter_estimation.py
│   ├── ou_simulation.py
│   ├── trading_strategy.py
│   ├── backtesting.py
│   ├── performance_metrics.py
│   └── sensitivity_analysis.py
│
├── experiments/
│   ├── parameter_validation.py
│   ├── simulation_experiments.py
│   └── robustness_experiments.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── trade_logs/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── report/
│
├── main.py
├── requirements.txt
└── README.md
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `data_loader.py` | Download and load historical prices |
| `preprocessing.py` | Align data, validate observations and calculate log prices |
| `pair_selection.py` | Correlation, OLS and ADF testing |
| `ou_model.py` | OU theoretical properties and mathematical functions |
| `parameter_estimation.py` | Exact AR(1)-based OU estimation |
| `ou_simulation.py` | Exact and Euler–Maruyama simulation |
| `trading_strategy.py` | Trading signals and position logic |
| `backtesting.py` | Historical portfolio simulation |
| `performance_metrics.py` | Financial performance and risk metrics |
| `sensitivity_analysis.py` | Parameter and robustness experiments |
| `main.py` | Run the complete pipeline |

---

## 7. Technology stack

Use the following Python libraries:

| Library | Purpose |
|---|---|
| Python | Main implementation language |
| NumPy | Numerical calculations and random simulation |
| Pandas | Financial time-series data |
| yfinance | Historical adjusted closing prices |
| SciPy | Numerical methods and optimisation |
| Statsmodels | OLS regression and ADF testing |
| Matplotlib | Plots |
| Seaborn | Statistical plots and heatmaps |

Keep dependencies in `requirements.txt`. Avoid introducing unnecessary packages.

---

## 8. Antigravity implementation workflow

Use Antigravity to implement the project phase by phase. Do not ask it to generate the entire project in one prompt.

For every phase:

1. Ask Antigravity to inspect the existing project and understand its current structure.
2. Identify the exact files that need to be created or modified.
3. Implement only the current phase.
4. Add appropriate validation and error handling.
5. Run the relevant code and tests.
6. Inspect generated outputs.
7. Independently verify important mathematical calculations.
8. Proceed to the next phase only after the current phase passes.

### Reusable base instruction for Antigravity

Use this instruction together with the detailed requirements for the current phase:

```text
We are implementing a Stochastic Calculus and Finance
project titled:

"Stochastic Modelling of Statistical Arbitrage Using
Ornstein-Uhlenbeck Processes: A Mean-Reversion Based
Pairs Trading Strategy."

Read the project implementation plan before making changes.

For the current phase:

1. Inspect the existing project structure.
2. Understand existing code and outputs.
3. Identify the exact files to create or modify.
4. Follow the mathematical formulation in the plan.
5. Use Python and the specified scientific libraries.
6. Write modular, readable and properly commented code.
7. Include appropriate validation and error handling.
8. Avoid look-ahead bias in all financial calculations.
9. Save all required numerical outputs and plots.
10. Add tests for important calculations.

Do not:
- Modify unrelated files.
- Duplicate existing functionality.
- Change the mathematical model without approval.
- Introduce unnecessary features.
- Silently ignore failed estimates or invalid data.
- Claim that an experiment passed without running it.

After implementation, report:
1. Files created or modified.
2. What was implemented.
3. Mathematical equations used.
4. How to execute the code.
5. Outputs generated.
6. Tests performed and their actual results.
7. Any known limitations or unresolved issues.
```

### Recommended implementation order

| Order | Implementation task |
|---:|---|
| 1 | Data loading and preprocessing |
| 2 | Pair selection and cointegration |
| 3 | OU model and theoretical properties |
| 4 | Parameter estimation |
| 5 | Simulation and estimator validation |
| 6 | Trading signals |
| 7 | Backtesting |
| 8 | Performance metrics |
| 9 | Sensitivity and robustness |
| 10 | Main pipeline, README and final verification |

---

## 9. Important mathematical and implementation checks

Use this as a verification checklist throughout development.

| Potential issue | Required check |
|---|---|
| Incorrect OU estimation | Validate using simulated data with known parameters |
| Incorrect half-life | Convert annualised half-life to trading days |
| Invalid AR(1) coefficient | Check \(0<b<1\) before parameter recovery |
| Incorrect transaction costs | Charge both legs on entry and exit |
| Look-ahead bias | Use only information available at each decision time |
| Incorrect pair selection | Use training data only |
| Overfitting | Do not tune using final test-period performance |
| Incorrect portfolio returns | Document hedge weights and position sizing |
| Structural breaks | Monitor rolling stationarity and half-life |
| Unrealistic trading assumptions | Document execution, short-selling and capital assumptions |

### Important modelling decisions

Preserve the following proposal choices in the initial implementation:

- Estimate the hedge ratio \(\beta\) using OLS on the training period.
- Use the resulting spread for OU modelling.
- Estimate OU parameters with a rolling 120-trading-day window.
- Use \(\Delta=1/252\).
- Use benchmark entry/exit levels of 2 and 0.5.
- Use a stop-loss at \(|s_t|=3.5\).
- Use transaction costs of 10 bps per leg.
- Evaluate performance on the specified out-of-sample period.

If any implementation detail is ambiguous, document the chosen convention rather than silently changing the proposal.

---

## 10. Final report structure (3–4 pages)

Keep the report concise and focused on mathematical formulation, implementation and findings.

| Section | Content | Approximate length |
|---|---|---:|
| 1. Introduction | Problem, motivation and objectives | ½ page |
| 2. Mathematical formulation | OU SDE, explicit solution, moments, stationary distribution, half-life and exact discretisation | 1 page |
| 3. Numerical methodology | Dataset, pair selection, parameter estimation, simulation and backtesting | 1 page |
| 4. Results and analysis | Simulation validation, strategy performance and sensitivity analysis | 1 page |
| 5. Conclusion | Main findings, limitations and future improvements | ½ page |

Include only the most informative figures and tables. Cite all models, algorithms and datasets.

---

## 11. Viva / presentation plan (15–20 minutes)

| Duration | Topic |
|---|---|
| 2 minutes | Problem statement and motivation |
| 4 minutes | OU mathematical formulation |
| 3 minutes | Parameter estimation and numerical simulation |
| 4 minutes | Trading strategy and backtesting |
| 3 minutes | Results and sensitivity analysis |
| 2 minutes | Limitations and conclusion |

Every group member should understand the complete pipeline, even if implementation responsibilities are divided.

Prepare to explain:

1. Why is an OU process appropriate for a mean-reverting spread?
2. Why is cointegration important in pairs trading?
3. How are \(\theta,\mu,\sigma\) estimated?
4. What is the difference between exact discretisation and Euler–Maruyama?
5. What does half-life mean financially?
6. How are trading signals generated?
7. How are transaction costs included?
8. Why can a historically cointegrated pair stop being suitable?
9. What do the sensitivity experiments reveal?
10. What are the limitations of the OU assumption?

---

## 12. Execution timeline

The target schedule below is designed around the 15 November 2026 deadline.

| Date | Target |
|---|---|
| 5 November | Data preparation and pair selection |
| 6 November | OU model and mathematical implementation |
| 7 November | Parameter estimation |
| 8 November | OU simulation and estimator validation |
| 9 November | Trading signals and backtesting |
| 10 November | Performance metrics and benchmark comparison |
| 11 November | Sensitivity analysis |
| 12 November | Structural-break analysis and robustness |
| 13 November | Final plots, tables and report |
| 14 November | Viva preparation and code verification |
| 15 November | Final submission |

If implementation takes longer than expected, prioritise:
1. Correct OU parameter estimation.
2. Simulation validation.
3. A correctly implemented out-of-sample backtest.
4. Performance and sensitivity analysis.
5. Report and viva preparation.

Do not sacrifice mathematical correctness to add extra features.

---

## 13. Final submission checklist

### Source code
- [ ] All Python source files are organised and commented.
- [ ] Dependencies are listed in `requirements.txt`.
- [ ] Code runs from the main entry point.
- [ ] Important mathematical calculations have tests.
- [ ] Outputs are reproducible.

### Mathematical formulation
- [ ] OU SDE and explicit solution are implemented/explained.
- [ ] Theoretical expectation and variance are verified.
- [ ] Stationary distribution is explained.
- [ ] Half-life is calculated with correct units.
- [ ] Exact discretisation is implemented.
- [ ] OU parameter estimation is validated.

### Numerical experiments
- [ ] Exact and Euler–Maruyama simulations are compared.
- [ ] Empirical and theoretical moments are compared.
- [ ] Pair selection is completed using training data.
- [ ] Out-of-sample backtesting is completed.
- [ ] Transaction costs and stop-losses are included.
- [ ] All required performance metrics are calculated.
- [ ] Sensitivity analysis is completed.
- [ ] Structural-break analysis is completed.
- [ ] Results are interpreted, not merely displayed.

### Documentation and presentation
- [ ] 3–4 page report is complete.
- [ ] Mathematical models, algorithms and datasets are referenced.
- [ ] Important figures and tables are included.
- [ ] Limitations are discussed.
- [ ] 15–20 minute viva presentation is prepared.
- [ ] All group members can explain the mathematical model and key findings.

---

## 14. Expected final project outcome

The completed project should demonstrate:

1. A mathematical treatment of the OU process and its theoretical properties.
2. A numerical method for estimating OU parameters from financial data.
3. Validation of the estimator using simulated data with known parameters.
4. A comparison of exact discretisation and Euler–Maruyama methods.
5. A pairs-trading strategy based on a cointegrated spread.
6. An out-of-sample evaluation using Nifty Bank constituent data.
7. A comparison against fixed z-score trading rules.
8. Sensitivity and robustness analysis showing when the strategy works and when it fails.

**Overall objective:** Demonstrate a clear and defensible connection between stochastic calculus, numerical computation and a real financial-market application.

The quality of the project should be judged not only by whether the strategy makes money, but also by whether the mathematical model is correctly formulated, numerically validated and critically interpreted.
