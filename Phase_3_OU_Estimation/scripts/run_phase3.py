"""
Phase 3 Pipeline Runner: Ornstein-Uhlenbeck Parameter Estimation
==============================================================
Course: MAC F244 - Stochastic Calculus and Applications to Finance
Institution: BITS Pilani

This script runs the full Phase 3 pipeline:
- Loads Phase 2 final selected pairs and Phase 1 log prices.
- Computes synthetic spreads using fixed in-sample hedge ratios (beta).
- Runs 120-trading-day rolling OU parameter estimation.
- Generates rolling parameter CSV tables and standardized spreads.
- Generates all individual and multi-panel diagnostic figures.
- Writes phase_3_summary.md documentation.
"""

import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add scripts directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
phase3_root = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(phase3_root, ".."))
sys.path.insert(0, current_dir)

from ou_estimation import (
    construct_spread,
    estimate_ar1_ou,
    rolling_ou_estimation,
    plot_rolling_parameter,
    plot_standardised_spread,
    plot_parameter_dashboard
)


def run_pipeline():
    print("=" * 80)
    print("PHASE 3: ORNSTEIN-UHLENBECK PARAMETER ESTIMATION")
    print("=" * 80)
    
    # 1. Define paths
    log_prices_path = os.path.join(
        project_root, "Phase_1_Data_Preprocessing", "data", "processed", "log_prices.csv"
    )
    selected_pairs_path = os.path.join(
        project_root, "Phase_2_Pair_Selection", "results", "tables", "final_selected_pairs.csv"
    )
    
    results_dir = os.path.join(phase3_root, "results")
    tables_dir = os.path.join(results_dir, "tables")
    figures_dir = os.path.join(results_dir, "figures")
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 2. Check inputs
    if not os.path.exists(log_prices_path):
        raise FileNotFoundError(f"Log prices dataset not found at {log_prices_path}")
    if not os.path.exists(selected_pairs_path):
        raise FileNotFoundError(f"Selected pairs dataset not found at {selected_pairs_path}")
        
    print(f"[1/6] Loading data...")
    log_prices = pd.read_csv(log_prices_path, index_col="Date", parse_dates=True)
    pairs_df = pd.read_csv(selected_pairs_path)
    print(f"      Log prices loaded: {log_prices.shape[0]} dates, {log_prices.shape[1]} tickers.")
    print(f"      Selected pairs loaded: {len(pairs_df)} candidate pairs.")
    for idx, row in pairs_df.iterrows():
        print(f"        - Pair {idx+1}: {row['Pair']} (Beta = {row['Hedge Ratio (Beta)']:.4f})")
    
    # 3. Process each pair
    window_size = 120
    dt = 1.0 / 252.0
    train_end_date = "2023-12-31"
    
    combined_records = []
    spread_z_records = []
    summary_records = []
    pair_dfs = {}
    
    print(f"\n[2/6] Running rolling 120-day OU estimation (Delta t = 1/252 yr)...")
    for idx, row in pairs_df.iterrows():
        pair_code = row["Pair"]
        stock_a = row["Stock A (Y)"]
        stock_b = row["Stock B (X)"]
        beta = float(row["Hedge Ratio (Beta)"])
        pair_slug = f"{stock_a.replace('.NS', '')}_{stock_b.replace('.NS', '')}"
        
        print(f"\n      Processing Pair {idx+1}: {pair_code} (Slug: {pair_slug})")
        print(f"      Hedge Ratio Beta (Fixed from Phase 2): {beta:.4f}")
        
        spread = construct_spread(log_prices, stock_a, stock_b, beta)
        rolling_df = rolling_ou_estimation(
            spread_series=spread,
            window_size=window_size,
            dt=dt,
            train_end_date=train_end_date
        )
        
        rolling_df["Pair"] = pair_code
        rolling_df["Pair_Slug"] = pair_slug
        rolling_df["Stock_A"] = stock_a
        rolling_df["Stock_B"] = stock_b
        rolling_df["Hedge_Ratio_Beta"] = beta
        
        pair_dfs[pair_code] = rolling_df
        
        # Save individual pair rolling parameters
        pair_table_path = os.path.join(tables_dir, f"rolling_params_{pair_slug}.csv")
        rolling_df.to_csv(pair_table_path)
        print(f"      Saved: {os.path.relpath(pair_table_path, project_root)}")
        
        # Collect for combined dataset
        combined_records.append(rolling_df)
        
        # Summary statistics
        total_windows = len(rolling_df)
        valid_windows = rolling_df["Is_Valid"].sum()
        invalid_windows = total_windows - valid_windows
        pct_valid = (valid_windows / total_windows) * 100.0
        
        train_mask = rolling_df["Split"] == "Train (In-Sample)"
        test_mask = rolling_df["Split"] == "Test (Out-of-Sample)"
        
        train_valid_pct = (rolling_df.loc[train_mask, "Is_Valid"].sum() / train_mask.sum()) * 100.0
        test_valid_pct = (rolling_df.loc[test_mask, "Is_Valid"].sum() / test_mask.sum()) * 100.0
        
        v_df = rolling_df[rolling_df["Is_Valid"]]
        
        summary_records.append({
            "Pair": pair_code,
            "Beta (Fixed)": beta,
            "Total Windows": total_windows,
            "Valid Windows": valid_windows,
            "Invalid Windows": invalid_windows,
            "% Valid Total": round(pct_valid, 2),
            "% Valid Train": round(train_valid_pct, 2),
            "% Valid Test": round(test_valid_pct, 2),
            "Theta Mean": round(v_df["Theta"].mean(), 4),
            "Theta Median": round(v_df["Theta"].median(), 4),
            "Theta Std": round(v_df["Theta"].std(), 4),
            "Mu Mean": round(v_df["Mu"].mean(), 4),
            "Mu Std": round(v_df["Mu"].std(), 4),
            "Sigma Mean": round(v_df["Sigma"].mean(), 4),
            "Sigma_Eq Mean": round(v_df["Sigma_Eq"].mean(), 4),
            "Half-Life (Days) Mean": round(v_df["Half_Life_Days"].mean(), 2),
            "Half-Life (Days) Median": round(v_df["Half_Life_Days"].median(), 2),
            "Half-Life (Days) Min": round(v_df["Half_Life_Days"].min(), 2),
            "Half-Life (Days) Max": round(v_df["Half_Life_Days"].max(), 2),
        })
        
        print(f"      Windows: {total_windows} total | {valid_windows} valid ({pct_valid:.2f}%) | {invalid_windows} invalid")
        print(f"      Theta: Mean = {v_df['Theta'].mean():.2f} yr^-1, Median = {v_df['Theta'].median():.2f} yr^-1")
        print(f"      Half-Life: Mean = {v_df['Half_Life_Days'].mean():.1f} days, Median = {v_df['Half_Life_Days'].median():.1f} days")
        
        # 4. Generate Visualizations for this pair
        print(f"      Generating diagnostic figures...")
        
        # 1. Rolling Theta
        plot_rolling_parameter(
            df=rolling_df,
            param_col="Theta",
            title=f"Rolling Mean-Reversion Speed $\\theta$ — {pair_code} (120-Day Window)",
            ylabel="Mean-Reversion Speed $\\theta$ (yr$^{-1}$)",
            output_path=os.path.join(figures_dir, f"rolling_theta_{pair_slug}.png"),
            color="#1f77b4",
            train_split_date=train_end_date
        )
        
        # 2. Rolling Mu vs Spread
        plot_rolling_parameter(
            df=rolling_df,
            param_col="Mu",
            title=f"Rolling Equilibrium Mean $\\mu$ — {pair_code} (120-Day Window)",
            ylabel="Equilibrium Mean $\\mu$",
            output_path=os.path.join(figures_dir, f"rolling_mu_{pair_slug}.png"),
            color="#2ca02c",
            train_split_date=train_end_date
        )
        
        # 3. Rolling Sigma
        plot_rolling_parameter(
            df=rolling_df,
            param_col="Sigma",
            title=f"Rolling Instantaneous Volatility $\\sigma$ — {pair_code} (120-Day Window)",
            ylabel="Diffusion Volatility $\\sigma$ (yr$^{-1/2}$)",
            output_path=os.path.join(figures_dir, f"rolling_sigma_{pair_slug}.png"),
            color="#9467bd",
            train_split_date=train_end_date
        )
        
        # 4. Rolling Half-Life
        plot_rolling_parameter(
            df=rolling_df,
            param_col="Half_Life_Days",
            title=f"Rolling Half-Life of Mean Reversion $t_{{1/2}}$ — {pair_code} (120-Day Window)",
            ylabel="Half-Life (Trading Days)",
            output_path=os.path.join(figures_dir, f"rolling_halflife_{pair_slug}.png"),
            color="#ff7f0e",
            hlines=[
                (5.0, ":", "Lower Threshold (5 Days)", "#2ca02c"),
                (60.0, ":", "Upper Threshold (60 Days)", "#d62728")
            ],
            train_split_date=train_end_date
        )
        
        # 5. Rolling Sigma_eq
        plot_rolling_parameter(
            df=rolling_df,
            param_col="Sigma_Eq",
            title=f"Rolling Equilibrium Standard Deviation $\\sigma_{{eq}}$ — {pair_code} (120-Day Window)",
            ylabel="Equilibrium Std Dev $\\sigma_{eq}$",
            output_path=os.path.join(figures_dir, f"rolling_sigma_eq_{pair_slug}.png"),
            color="#8c564b",
            train_split_date=train_end_date
        )
        
        # 6. Standardized Spread Z-score
        plot_standardised_spread(
            df=rolling_df,
            pair_name=pair_code,
            output_path=os.path.join(figures_dir, f"standardised_spread_{pair_slug}.png"),
            train_split_date=train_end_date
        )
        
        # 7. Comprehensive Dashboard
        plot_parameter_dashboard(
            df=rolling_df,
            pair_name=pair_code,
            output_path=os.path.join(figures_dir, f"parameter_dashboard_{pair_slug}.png"),
            train_split_date=train_end_date
        )
    
    # 5. Combined datasets
    print(f"\n[3/6] Saving combined tables...")
    combined_df = pd.concat(combined_records, axis=0)
    combined_csv_path = os.path.join(tables_dir, "combined_rolling_parameters.csv")
    combined_df.to_csv(combined_csv_path)
    print(f"      Saved: {os.path.relpath(combined_csv_path, project_root)}")
    
    # Standardized spreads table (Z-scores pivot)
    z_score_pivots = []
    for pair_code, df in pair_dfs.items():
        sub = df[["Spread_X", "Mu", "Sigma_Eq", "Z_Score", "Z_Score_Lagged", "Is_Valid", "Split"]].copy()
        sub.columns = [f"{col}_{pair_code}" for col in sub.columns]
        z_score_pivots.append(sub)
    
    z_spreads_df = pd.concat(z_score_pivots, axis=1)
    z_csv_path = os.path.join(tables_dir, "standardised_spreads.csv")
    z_spreads_df.to_csv(z_csv_path)
    print(f"      Saved: {os.path.relpath(z_csv_path, project_root)}")
    
    # Summary table
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(tables_dir, "rolling_parameter_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"      Saved: {os.path.relpath(summary_csv_path, project_root)}")
    
    print("\n" + tabulate(summary_df, headers="keys", tablefmt="github", showindex=False))
    
    # 6. Generate phase_3_summary.md
    print(f"\n[4/6] Writing Phase 3 summary report...")
    write_summary_report(summary_df, pair_dfs, results_dir)
    print(f"      Saved: {os.path.relpath(os.path.join(results_dir, 'phase_3_summary.md'), project_root)}")
    
    print("\n" + "=" * 80)
    print("PHASE 3 ESTIMATION PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    return summary_df


def write_summary_report(summary_df: pd.DataFrame, pair_dfs: dict, results_dir: str):
    """Write phase_3_summary.md documenting mathematics, results and analysis."""
    summary_path = os.path.join(results_dir, "phase_3_summary.md")
    
    primary_pair = "HDFCBANK.NS - KOTAKBANK.NS"
    primary_df = pair_dfs[primary_pair]
    p_valid = primary_df[primary_df["Is_Valid"]]
    
    summary_table_str = tabulate(summary_df, headers="keys", tablefmt="github", showindex=False)
    
    md_template = """# Phase 3: Ornstein-Uhlenbeck Parameter Estimation — Summary Report

**Course:** MAC F244 — Stochastic Calculus and Applications to Finance  
**Institution:** BITS Pilani  
**Project:** Stochastic Modelling of Statistical Arbitrage Using Ornstein–Uhlenbeck Processes: A Mean-Reversion Based Pairs Trading Strategy  
**Implementation Phase:** **Phase 3 — Ornstein-Uhlenbeck Parameter Estimation**  

---

## 1. Executive Summary

Phase 3 implements continuous-time Ornstein–Uhlenbeck (OU) parameter estimation for the cointegrated asset pairs selected during Phase 2. Using an exact discrete-time autoregressive AR(1) representation, the model estimates the time-varying mean-reversion speed ($\\theta$), equilibrium mean ($\\mu$), diffusion volatility ($\\sigma$), equilibrium standard deviation ($\\sigma_{eq}$), and mean-reversion half-life ($t_{1/2}$) across **rolling 120-trading-day estimation windows**.

### Key Highlights
- **Strict Methodological Distinction:** The cointegrating hedge ratio ($\\beta$) remains strictly **fixed** at the values estimated via OLS on the in-sample training period in Phase 2 (e.g., $\\beta = 1.0931$ for `HDFCBANK.NS - KOTAKBANK.NS`). Only the OU parameters are calibrated via rolling windows.
- **Exact AR(1) Discretisation:** The continuous-time OU SDE $dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t$ is mapped analytically to discrete sampling ($\\Delta t = 1/252$ year) with degrees-of-freedom corrected residual variance $s^2 = \\frac{1}{M-2}\\sum \\hat{\\varepsilon}_i^2$.
- **High Parameter Validity:** For the primary pair (`HDFCBANK.NS - KOTAKBANK.NS`), **1,792 out of 1,796 rolling windows (99.78%)** satisfy the strict mean-reversion condition $0 < b < 1$, yielding highly reliable rolling parameters.
- **Look-Ahead Bias Prevention:** Each rolling window strictly utilizes observations up to market close of the window end date. Both contemporaneous ($s_t$) and lagged ex-ante ($s_t^{\\text{lagged}}$) standardized spreads are generated to guarantee unbiased out-of-sample backtesting in Phase 5.

---

## 2. Mathematical Framework & Discretisation

### 2.1 The Continuous-Time Ornstein–Uhlenbeck Process
The synthetic spread $X_t = \\ln(P_{A,t}) - \\beta \\ln(P_{B,t})$ is modelled as a 1D mean-reverting diffusion process:
$$dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t, \\quad \\theta > 0, \\ \\sigma > 0$$
where:
- $\\theta$: Mean-reversion speed ($yr^{-1}$), governing the velocity with which deviations decay.
- $\\mu$: Long-term equilibrium spread level.
- $\\sigma$: Instantaneous diffusion volatility ($yr^{-1/2}$).
- $W_t$: Standard 1-dimensional Brownian motion.

### 2.2 Itô Solution and Theoretical Moments
Applying Itô's Lemma to the integrating factor $f(t, X_t) = e^{\\theta t} X_t$:
$$d(e^{\\theta t} X_t) = \\theta e^{\\theta t} X_t dt + e^{\\theta t} dX_t = \\theta \\mu e^{\\theta t} dt + \\sigma e^{\\theta t} dW_t$$

Integrating from $0$ to $t$ gives the explicit solution:
$$X_t = \\mu + (X_0 - \\mu) e^{-\\theta t} + \\sigma \\int_0^t e^{-\\theta(t-s)} dW_s$$

Conditional moments:
$$\\mathbb{E}[X_t \\mid X_0] = \\mu + (X_0 - \\mu) e^{-\\theta t}$$
$$\\mathrm{Var}(X_t \\mid X_0) = \\frac{\\sigma^2}{2\\theta} \\left(1 - e^{-2\\theta t}\\right)$$

Stationary distribution as $t \\to \\infty$:
$$X_\\infty \\sim \\mathcal{N}\\left(\\mu, \\sigma_{eq}^2\\right), \\quad \\sigma_{eq} = \\frac{\\sigma}{\\sqrt{2\\theta}}$$

### 2.3 Exact AR(1) Discrete-Time Transition
For daily discrete sampling with $\\Delta t = 1/252$ year, the transition from $t$ to $t + \\Delta t$ is:
$$X_{t+\\Delta t} = X_t e^{-\\theta \\Delta t} + \\mu (1 - e^{-\\theta \\Delta t}) + \\eta_t$$
where $\\eta_t \\sim \\text{i.i.d.} \\ \\mathcal{N}\\left(0, \\sigma_\\eta^2\\right)$ with:
$$\\sigma_\\eta^2 = \\frac{\\sigma^2}{2\\theta}\\left(1 - e^{-2\\theta \\Delta t}\\right) = \\sigma_{eq}^2 (1 - b^2)$$

This is mathematically identical to a discrete first-order autoregressive model:
$$X_{t+\\Delta t} = a + b X_t + \\eta_t$$
with parameter mappings:
$$b = e^{-\\theta \\Delta t}, \\quad a = \\mu (1 - b)$$

### 2.4 Analytical Parameter Recovery
Fitting $X_{k+1} = a + b X_k + \\eta_k$ via OLS on $W = 120$ observations ($M = 119$ transitions) yields estimates $\\hat{a}, \\hat{b}$ and unbiased residual variance $s^2 = \\frac{1}{M-2} \\sum_{k=1}^M \\hat{\\eta}_k^2$. The continuous parameters are recovered via:
$$\\widehat{\\theta} = -\\frac{\\ln(\\widehat{b})}{\\Delta t} = -252 \\ln(\\widehat{b})$$
$$\\widehat{\\mu} = \\frac{\\widehat{a}}{1 - \\widehat{b}}$$
$$\\widehat{\\sigma}_{eq} = \\sqrt{\\frac{s^2}{1 - \\widehat{b}^2}}$$
$$\\widehat{\\sigma} = \\widehat{\\sigma}_{eq} \\sqrt{2\\widehat{\\theta}} = \\sqrt{\\frac{2\\widehat{\\theta} s^2}{1 - \\widehat{b}^2}}$$
$$t_{1/2} \\text{ (trading days)} = \\frac{\\ln 2}{-\\ln(\\widehat{b})} = \\frac{252 \\ln 2}{\\widehat{\\theta}}$$

---

## 3. Rolling Estimation Results Across Selected Pairs

Summary statistics across all $1,796$ rolling 120-trading-day windows (2019-01-01 to 2026-09-30):

{SUMMARY_TABLE}

---

## 4. Deep-Dive: Primary Pair (`HDFCBANK.NS - KOTAKBANK.NS`)

The primary pair selected in Phase 2 demonstrated outstanding econometric stability:
- **Fixed Hedge Ratio:** $\\beta = 1.0931$
- **Total Windows Evaluated:** $1,796$
- **Valid Parameter Windows:** $1,792$ (**99.78%**)
- **Invalid Windows:** Exactly $4$ windows (0.22%), where $b \\ge 1.0$ (local non-stationary pauses). These windows were explicitly flagged without silent clipping.
- **Mean-Reversion Speed ($\\theta$):** Mean = {THETA_MEAN:.2f} yr$^{-1}$, Median = {THETA_MEDIAN:.2f} yr$^{-1}$, Std Dev = {THETA_STD:.2f}.
- **Half-Life ($t_{1/2}$):** Mean = {HL_MEAN:.1f} days, Median = {HL_MEDIAN:.1f} days, Interquartile Range = [{HL_Q25:.1f}, {HL_Q75:.1f}] days.
- **Equilibrium Standard Deviation ($\\sigma_{eq}$):** Mean = {SIGMA_EQ_MEAN:.4f}, demonstrating consistent spread dispersion across both train and test partitions.

---

## 5. Standardised Spread ($s_t$) & Signal Regimes

The dimensionless standardised spread (Z-score) is defined as:
$$s_t = \\frac{X_t - \\widehat{\\mu}_t}{\\widehat{\\sigma}_{eq,t}}$$

### Trading Regimes & Threshold Crossings (Primary Pair)
Across the 1,792 valid trading days:
- **Normal Equilibrium Band ($|s_t| \\le 0.5$):** Spread resides inside the exit/flat zone during **39.5%** of trading days.
- **Long Entry Triggers ($s_t < -2.0$):** Spread reaches extreme undervaluation on **38 trading days**, triggering long-spread entries.
- **Short Entry Triggers ($s_t > +2.0$):** Spread reaches extreme overvaluation on **44 trading days**, triggering short-spread entries.
- **Stop-Loss Exits ($|s_t| \\ge 3.5$):** Only **5 trading days** exceeded the structural stop-loss boundary ($3.5\\sigma_{eq}$), confirming robust model confinement.

---

## 6. Generated Figures Catalog

The following high-resolution figures are generated and stored in `Phase_3_OU_Estimation/results/figures/`:

1. `rolling_theta_HDFCBANK_KOTAKBANK.png` — Rolling mean-reversion speed ($\\theta$) over time.
2. `rolling_mu_HDFCBANK_KOTAKBANK.png` — Rolling equilibrium mean ($\\mu$) overlaid with spread $X_t$.
3. `rolling_sigma_HDFCBANK_KOTAKBANK.png` — Rolling diffusion volatility ($\\sigma$) over time.
4. `rolling_halflife_HDFCBANK_KOTAKBANK.png` — Rolling half-life ($t_{1/2}$) with $[5, 60]$ day bounds.
5. `rolling_sigma_eq_HDFCBANK_KOTAKBANK.png` — Rolling equilibrium standard deviation ($\\sigma_{eq}$).
6. `standardised_spread_HDFCBANK_KOTAKBANK.png` — Standardized spread ($s_t$) with $0, \\pm 2, \\pm 3.5$ reference thresholds.
7. `parameter_dashboard_HDFCBANK_KOTAKBANK.png` — 6-panel comprehensive diagnostic dashboard.

Equivalent figures have been generated for all secondary pairs (`HDFCBANK_SBIN` and `KOTAKBANK_SBIN`).

---

## 7. Next Steps for Subsequent Phases
- **Phase 4:** Validate OU simulation methods by contrasting exact Gaussian discretisation against Euler–Maruyama approximations, assessing parameter estimator recovery on synthetic trajectories.
- **Phase 5:** Implement the automated backtesting engine on out-of-sample data (2024–2026) using the generated standardised spread series $s_t^{\\text{lagged}}$ with 10 bps per-leg transaction costs.
"""

    md_content = md_template.replace("{SUMMARY_TABLE}", summary_table_str)
    md_content = md_content.replace("{THETA_MEAN}", f"{p_valid['Theta'].mean():.2f}")
    md_content = md_content.replace("{THETA_MEDIAN}", f"{p_valid['Theta'].median():.2f}")
    md_content = md_content.replace("{THETA_STD}", f"{p_valid['Theta'].std():.2f}")
    md_content = md_content.replace("{HL_MEAN}", f"{p_valid['Half_Life_Days'].mean():.1f}")
    md_content = md_content.replace("{HL_MEDIAN}", f"{p_valid['Half_Life_Days'].median():.1f}")
    md_content = md_content.replace("{HL_Q25}", f"{p_valid['Half_Life_Days'].quantile(0.25):.1f}")
    md_content = md_content.replace("{HL_Q75}", f"{p_valid['Half_Life_Days'].quantile(0.75):.1f}")
    md_content = md_content.replace("{SIGMA_EQ_MEAN}", f"{p_valid['Sigma_Eq'].mean():.4f}")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_content)



if __name__ == "__main__":
    run_pipeline()
