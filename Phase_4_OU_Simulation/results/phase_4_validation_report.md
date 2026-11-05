# Phase 4: OU Process Simulation and Numerical Validation — Validation Report

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Institution:** BITS Pilani  
**Audit Date:** 2026-11-05  
**Audit Scope:** Independent mathematical, algorithmic, and numerical verification of Phase 4 OU simulations.

---

## 1. Audit Executive Summary

An independent 10-point audit was performed on all simulation algorithms, discretisation routines, stability criteria, theoretical moment equations, and error calculations in Phase 4.

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

| Check                                     | Requirement                                                                     | Observed                                             | Status   |
|-------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------|----------|
| 1. Exact OU Transition Formula            | X_(t+dt) = mu*(1-e^(-th*dt)) + e^(-th*dt)*X_t + sigma_eta*Z exactly verified    | Max difference: 0.00e+00                             | PASSED   |
| 2. Euler-Maruyama Formula                 | X_(t+dt) = X_t + th*(mu - X_t)*dt + sig*sqrt(dt)*Z exactly verified             | Max difference: 1.11e-16                             | PASSED   |
| 3. Coupled Randomness Consistency         | Both schemes driven by identical Z_t; error converges toward zero as dt -> 0    | Max coupled diff: 0.0001 at fine dt                  | PASSED   |
| 4. Euler Stability Criteria               | Euler stable if theta*dt < 2; explosive if theta*dt >= 2 correctly detected     | Daily stable: True, Supercritical: False             | PASSED   |
| 5. Theoretical Mean Formula               | E[X_t] = mu + (X_0 - mu)*exp(-theta*t) verified                                 | Max difference: 0.00e+00                             | PASSED   |
| 6. Theoretical Variance Formula           | Var(X_t) = (sigma^2 / (2*theta)) * (1 - exp(-2*theta*t)) verified               | Max difference: 0.00e+00                             | PASSED   |
| 7. Stationary Variance Identity           | sigma_eq^2 = sigma^2 / (2*theta), sigma_eq = sigma / sqrt(2*theta) verified     | Variance: 0.002000, sigma_eq: 0.044721               | PASSED   |
| 8. MAE & RMSE Error Metrics               | Mathematical definitions of pathwise MAE and RMSE verified                      | MAE=0.2333, RMSE=0.2380 match definitions            | PASSED   |
| 9. Parameter Recovery Consistency         | AR(1) OLS and continuous OU recovery use identical formulas as Phase 3          | Recovered theta = -ln(b)/dt exactly matches Phase 3  | PASSED   |
| 10. Manually Verifiable Numerical Example | Analytical benchmark test with theta=10, mu=0.50, sigma=0.20, dt=1/252 verified | b=0.961095, a=0.019453, sig_eta=0.012353, g=0.960317 | PASSED   |

---

## 3. Detailed Audit Findings

### 3.1 Exact OU Discretisation vs Euler–Maruyama Formulations
1. **Exact Discretisation:**
   $$X_{t+\Delta t} = \mu(1 - e^{-\theta \Delta t}) + e^{-\theta \Delta t} X_t + \sigma \sqrt{\frac{1 - e^{-2\theta \Delta t}}{2\theta}} Z_t$$
   Verified against 1-step analytical propagation with zero numerical discrepancies ($< 10^{-12}$).
2. **Euler–Maruyama Approximation:**
   $$X_{t+\Delta t} = X_t + \theta(\mu - X_t)\Delta t + \sigma \sqrt{\Delta t} Z_t = (1 - \theta \Delta t)X_t + \theta \mu \Delta t + \sigma \sqrt{\Delta t} Z_t$$
   Verified against 1-step analytical propagation ($< 10^{-12}$).

### 3.2 Coupled Randomness Verification
Coupled path simulation guarantees that both schemes receive identical underlying standard Gaussian increments $Z_t$. As $\Delta t \to 0$, the pathwise error between exact and Euler approaches zero ($< 0.005$ at fine resolution), confirming that path differences isolate pure discretisation error without Monte Carlo noise contamination.

### 3.3 Linear Stability Analysis
The linear stability boundary $\Delta t < \frac{2}{\theta}$ was verified:
- For daily trading frequency ($\Delta t = 1/252$), $\theta \Delta t \ll 1$, ensuring monotonic stability.
- For supercritical step size ($\theta = 25.0, \Delta t = 0.25 \implies \theta \Delta t = 6.25 > 2$), the Euler scheme exhibited explosive numerical overflow ($|1 - \theta \Delta t| = 5.25 \gg 1$), while exact discretisation remained unconditionally stable ($b = e^{-6.25} = 0.0019 \in (0, 1)$).

### 3.4 Theoretical vs. Empirical Moments
Across $10,000$ independent simulated paths over $T = 3.0$ years:
- Empirical expectation $\hat{\mathbb{E}}[X_t]$ tracked $\mu + (X_0 - \mu)e^{-\theta t}$ across all time slices with relative error $< 0.15\%$.
- Empirical variance $\hat{\mathrm{Var}}(X_t)$ tracked $\frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$ and converged asymptotically to $\sigma_{eq}^2 = \frac{\sigma^2}{2\theta}$ within $0.4\%$ error.

---

## 4. Manually Verifiable Benchmark Example

**Parameters:**
- $\theta = 10.0\text{ yr}^{-1}$, $\mu = 0.50$, $\sigma = 0.20$, $X_0 = 1.00$, $\Delta t = 1/252\text{ yr} \approx 0.00396825$.

**Hand Calculations:**
1. Exact decay parameter:
   $$b = e^{-10 \times 0.00396825} = e^{-0.0396825} = \mathbf{0.961094}$$
2. Exact drift constant:
   $$a = 0.50 \times (1 - 0.961094) = \mathbf{0.019453}$$
3. Stationary variance:
   $$\sigma_{eq}^2 = \frac{0.20^2}{2 \times 10.0} = \mathbf{0.002000} \implies \sigma_{eq} = \mathbf{0.044721}$$
4. Exact innovation scale:
   $$\sigma_\eta = \sqrt{0.0020 \times (1 - 0.961094^2)} = \mathbf{0.012353}$$
5. Euler amplification factor:
   $$g = 1 - \theta \Delta t = 1 - 10 \times 0.00396825 = \mathbf{0.960317}$$

The codebase reproduces these exact values within numerical tolerance ($< 10^{-6}$).

---

## 5. Limitations & Methodological Constraints
1. **Finite Sample Bias (Hurwicz Bias):** In AR(1) parameter recovery experiments, the finite-sample Hurwicz bias in $\widehat{b}$ introduces a modest positive bias in recovered $\widehat{\theta}$ for small sample lengths ($N = 120$). This is an established econometric property of autoregressive models, not an algorithmic defect.
2. **Coupled Brownian Noise:** Pathwise coupling provides superior diagnostic precision for comparing discretisation methods, but in empirical financial markets, only discrete sampled prices are observable.
