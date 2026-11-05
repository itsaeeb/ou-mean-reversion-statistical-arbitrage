"""
Phase 3: Ornstein-Uhlenbeck Parameter Estimation Module
======================================================
Course: MAC F244 - Stochastic Calculus and Applications to Finance
Institution: BITS Pilani

This module implements:
1. Exact discrete-time AR(1) representation of the continuous-time OU process:
      dX_t = theta * (mu - X_t) * dt + sigma * dW_t
   Discretization (Delta = 1/252 yr):
      X_(t+Delta) = a + b * X_t + eta_t
      where b = exp(-theta * Delta), a = mu * (1 - b),
      Var(eta_t) = sigma^2 * (1 - exp(-2 * theta * Delta)) / (2 * theta)
2. Analytical recovery of OU parameters (theta, mu, sigma, sigma_eq, half-life).
3. Rolling 120-trading-day window estimation engine with rigorous validation flags:
   - Requirement: 0 < b < 1 for stationary mean-reversion.
   - Non-finite, explosive (b >= 1), or oscillatory (b <= 0) estimates explicitly flagged.
4. Calculation of equilibrium standard deviation sigma_eq = sigma / sqrt(2*theta).
5. Standardized spread (Z-score) series with look-ahead bias prevention.
6. Comprehensive visualization suite.
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Suppress warnings
warnings.filterwarnings("ignore")

# Visualization styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def construct_spread(log_prices: pd.DataFrame, stock_a: str, stock_b: str, beta: float) -> pd.Series:
    """
    Construct the synthetic spread X_t = ln(P_A,t) - beta * ln(P_B,t) using
    a fixed hedge ratio beta estimated from Phase 2 training data.
    """
    if stock_a not in log_prices.columns or stock_b not in log_prices.columns:
        raise ValueError(f"Stocks {stock_a} or {stock_b} not found in log prices.")
    
    spread = log_prices[stock_a] - beta * log_prices[stock_b]
    spread.name = f"Spread_{stock_a}_{stock_b}"
    return spread


def estimate_ar1_ou(window_values: np.ndarray, dt: float = 1.0 / 252.0) -> dict:
    """
    Fit an exact AR(1) model to a 1D window of spread observations and recover
    the continuous-time Ornstein-Uhlenbeck parameters.
    
    Parameters:
    -----------
    window_values : np.ndarray
        Array of spread values of length W (e.g., 120 trading days).
    dt : float
        Sampling interval in annual units (default 1/252 for daily data).
        
    Returns:
    --------
    dict containing:
        - ar1_a : Intercept of AR(1)
        - ar1_b : Autoregressive coefficient of AR(1)
        - residual_var : Unbiased sample variance of residuals (s^2 = SSE / (M - 2))
        - theta : Mean-reversion speed (annualized)
        - mu : Long-term equilibrium mean
        - sigma : Instantaneous diffusion volatility (annualized)
        - sigma_eq : Equilibrium standard deviation (sigma / sqrt(2*theta))
        - half_life_days : Half-life in trading days (ln(2) / (theta * dt))
        - half_life_years : Half-life in years (ln(2) / theta)
        - is_valid : Boolean validity flag
        - failure_reason : String description if invalid, "VALID" otherwise
    """
    N = len(window_values)
    if N < 5:
        return {
            "ar1_a": np.nan, "ar1_b": np.nan, "residual_var": np.nan,
            "theta": np.nan, "mu": np.nan, "sigma": np.nan, "sigma_eq": np.nan,
            "half_life_days": np.nan, "half_life_years": np.nan,
            "is_valid": False, "failure_reason": "Insufficient observations (< 5)"
        }
    
    # Form AR(1) pairs: Y_k = X_(k+1), X_k = X_k for k = 0, ..., N-2
    # Sample size for regression: M = N - 1
    Y = window_values[1:]
    X = window_values[:-1]
    M = len(Y)
    
    # OLS analytical estimation
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    x_dev = X - x_mean
    y_dev = Y - y_mean
    
    sum_xx = np.sum(x_dev ** 2)
    sum_xy = np.sum(x_dev * y_dev)
    
    if sum_xx <= 1e-14 or not np.isfinite(sum_xx):
        return {
            "ar1_a": np.nan, "ar1_b": np.nan, "residual_var": np.nan,
            "theta": np.nan, "mu": np.nan, "sigma": np.nan, "sigma_eq": np.nan,
            "half_life_days": np.nan, "half_life_years": np.nan,
            "is_valid": False, "failure_reason": "Zero or infinite variance in regressors"
        }
    
    b = sum_xy / sum_xx
    a = y_mean - b * x_mean
    
    residuals = Y - (a + b * X)
    sse = np.sum(residuals ** 2)
    
    # Degrees of freedom for AR(1) with intercept: M - 2 = N - 3
    dof = max(M - 2, 1)
    res_var = sse / dof
    
    # Check numerical validity
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(res_var)):
        return {
            "ar1_a": a, "ar1_b": b, "residual_var": res_var,
            "theta": np.nan, "mu": np.nan, "sigma": np.nan, "sigma_eq": np.nan,
            "half_life_days": np.nan, "half_life_years": np.nan,
            "is_valid": False, "failure_reason": "Non-finite AR(1) regression estimates"
        }
    
    if b >= 1.0:
        return {
            "ar1_a": a, "ar1_b": b, "residual_var": res_var,
            "theta": np.nan, "mu": np.nan, "sigma": np.nan, "sigma_eq": np.nan,
            "half_life_days": np.nan, "half_life_years": np.nan,
            "is_valid": False, "failure_reason": f"b >= 1 (b = {b:.6f}): Unit-root or explosive non-stationary drift"
        }
    
    if b <= 0.0:
        return {
            "ar1_a": a, "ar1_b": b, "residual_var": res_var,
            "theta": np.nan, "mu": np.nan, "sigma": np.nan, "sigma_eq": np.nan,
            "half_life_days": np.nan, "half_life_years": np.nan,
            "is_valid": False, "failure_reason": f"b <= 0 (b = {b:.6f}): Non-positive / oscillatory AR(1)"
        }
    
    if res_var <= 0.0:
        return {
            "ar1_a": a, "ar1_b": b, "residual_var": res_var,
            "theta": np.nan, "mu": np.nan, "sigma": np.nan, "sigma_eq": np.nan,
            "half_life_days": np.nan, "half_life_years": np.nan,
            "is_valid": False, "failure_reason": "Residual variance <= 0"
        }
    
    # Exact OU Parameter Recovery
    # theta = -ln(b) / dt
    theta = -np.log(b) / dt
    
    # mu = a / (1 - b)
    mu = a / (1.0 - b)
    
    # Equilibrium variance: sigma_eq^2 = res_var / (1 - b^2)
    # sigma^2 = 2 * theta * res_var / (1 - b^2)
    sigma_eq = np.sqrt(res_var / (1.0 - b ** 2))
    sigma = sigma_eq * np.sqrt(2.0 * theta)
    
    # Half-life: ln(2) / theta in years, and in trading days = ln(2) / (-ln(b))
    half_life_years = np.log(2.0) / theta
    half_life_days = half_life_years / dt  # = np.log(2.0) / (-np.log(b))
    
    return {
        "ar1_a": a,
        "ar1_b": b,
        "residual_var": res_var,
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "sigma_eq": sigma_eq,
        "half_life_days": half_life_days,
        "half_life_years": half_life_years,
        "is_valid": True,
        "failure_reason": "VALID"
    }


def rolling_ou_estimation(
    spread_series: pd.Series,
    window_size: int = 120,
    dt: float = 1.0 / 252.0,
    train_end_date: str = "2023-12-31"
) -> pd.DataFrame:
    """
    Perform rolling 120-trading-day OU parameter estimation on a spread series.
    
    Alignment & Look-Ahead Bias Prevention:
    ---------------------------------------
    - Window index i corresponds to observations [i - window_size, ..., i - 1].
    - Window start date: spread_series.index[i - window_size]
    - Window end date: spread_series.index[i - 1]
    - Parameter estimates available at the close of Window_End_Date are derived
      STRICTLY from data up to and including Window_End_Date.
    - Standardized spread (Contemporaneous):
        s_t = (X_t - mu_t) / sigma_eq,t
      using parameters estimated up to date t.
    - Standardized spread (Lagged / Strict ex-ante):
        s_t_lagged = (X_t - mu_(t-1)) / sigma_eq,(t-1)
      using parameters estimated strictly PRIOR to date t (from t-1 window).
    """
    N = len(spread_series)
    if N < window_size:
        raise ValueError(f"Spread length ({N}) is less than window size ({window_size}).")
    
    results = []
    values = spread_series.values
    dates = spread_series.index
    
    train_end_ts = pd.Timestamp(train_end_date)
    
    for i in range(window_size, N + 1):
        window_vals = values[i - window_size : i]
        start_date = dates[i - window_size]
        end_date = dates[i - 1]
        spread_val = values[i - 1]
        
        # Determine dataset partition
        split = "Train (In-Sample)" if end_date <= train_end_ts else "Test (Out-of-Sample)"
        
        # Estimate parameters
        res = estimate_ar1_ou(window_vals, dt=dt)
        
        # Contemporaneous Z-Score
        if res["is_valid"] and res["sigma_eq"] > 0:
            z_score = (spread_val - res["mu"]) / res["sigma_eq"]
        else:
            z_score = np.nan
        
        row = {
            "Window_Index": len(results),
            "Window_Start_Date": start_date,
            "Window_End_Date": end_date,
            "Date": end_date,
            "Spread_X": spread_val,
            "AR1_a": res["ar1_a"],
            "AR1_b": res["ar1_b"],
            "Residual_Var": res["residual_var"],
            "Theta": res["theta"],
            "Mu": res["mu"],
            "Sigma": res["sigma"],
            "Sigma_Eq": res["sigma_eq"],
            "Half_Life_Days": res["half_life_days"],
            "Half_Life_Years": res["half_life_years"],
            "Is_Valid": res["is_valid"],
            "Failure_Reason": res["failure_reason"],
            "Z_Score": z_score,
            "Split": split
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    df.set_index("Date", inplace=True)
    
    # Compute Lagged Z-score (ex-ante alignment)
    # Uses previous day's mu and sigma_eq for today's spread
    df["Mu_Lagged"] = df["Mu"].shift(1)
    df["Sigma_Eq_Lagged"] = df["Sigma_Eq"].shift(1)
    df["Z_Score_Lagged"] = np.where(
        df["Sigma_Eq_Lagged"] > 0,
        (df["Spread_X"] - df["Mu_Lagged"]) / df["Sigma_Eq_Lagged"],
        np.nan
    )
    
    return df


# =====================================================================
# Visualizations
# =====================================================================

def plot_rolling_parameter(
    df: pd.DataFrame,
    param_col: str,
    title: str,
    ylabel: str,
    output_path: str,
    color: str = "#1f77b4",
    hlines: list = None,
    train_split_date: str = "2023-12-31"
):
    """Plot a single rolling parameter series with train/test division."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    
    valid_mask = df["Is_Valid"]
    dates = df.index[valid_mask]
    series = df.loc[valid_mask, param_col]
    
    ax.plot(dates, series, label=f"Rolling {ylabel}", color=color, linewidth=1.5)
    
    # Train / test split line
    split_dt = pd.Timestamp(train_split_date)
    if split_dt in df.index or (split_dt >= df.index[0] and split_dt <= df.index[-1]):
        ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Train/Test Boundary ({train_split_date})")
    
    # Reference horizontal lines
    if hlines:
        for val, style, lbl, clr in hlines:
            ax.axhline(val, linestyle=style, color=clr, linewidth=1.2, label=lbl)
            
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_standardised_spread(
    df: pd.DataFrame,
    pair_name: str,
    output_path: str,
    train_split_date: str = "2023-12-31"
):
    """Plot standardized spread Z-score with entry/exit/stop-loss thresholds."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    valid_mask = df["Is_Valid"]
    dates = df.index[valid_mask]
    z_scores = df.loc[valid_mask, "Z_Score"]
    
    ax.plot(dates, z_scores, color="#2b5c8f", linewidth=1.4, label="Standardised Spread $s_t$ (Z-Score)")
    
    # Equilibrium mean line
    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0, label="Equilibrium ($s = 0$)")
    
    # Entry threshold lines (+/- 2.0)
    ax.axhline(2.0, color="#d95f02", linestyle="--", linewidth=1.2, label="Short Entry ($s = +2.0$)")
    ax.axhline(-2.0, color="#1b9e77", linestyle="--", linewidth=1.2, label="Long Entry ($s = -2.0$)")
    
    # Exit threshold region (+/- 0.5)
    ax.axhline(0.5, color="#7570b3", linestyle=":", linewidth=1.0, label="Exit Band ($|s| = 0.5$)")
    ax.axhline(-0.5, color="#7570b3", linestyle=":", linewidth=1.0)
    
    # Stop-loss lines (+/- 3.5)
    ax.axhline(3.5, color="#e7298a", linestyle="-.", linewidth=1.2, label="Stop-Loss ($|s| = 3.5$)")
    ax.axhline(-3.5, color="#e7298a", linestyle="-.", linewidth=1.2)
    
    # Train / test boundary
    split_dt = pd.Timestamp(train_split_date)
    ax.axvline(split_dt, color="#e41a1c", linestyle="-", linewidth=1.8,
               label=f"Train/Test Boundary ({train_split_date})")
    
    # Shade trading trigger zones
    ax.fill_between(dates, 2.0, 3.5, color="#d95f02", alpha=0.10, label="Short Entry Zone")
    ax.fill_between(dates, -3.5, -2.0, color="#1b9e77", alpha=0.10, label="Long Entry Zone")
    ax.fill_between(dates, -0.5, 0.5, color="#7570b3", alpha=0.10, label="Mean-Reversion Exit Zone")
    
    ax.set_title(f"Standardised Spread $s_t$ Dynamics — {pair_name} (120-Day Rolling Window)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Standardised Spread $s_t$", fontsize=11, fontweight="bold")
    ax.set_ylim(-4.5, 4.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_parameter_dashboard(
    df: pd.DataFrame,
    pair_name: str,
    output_path: str,
    train_split_date: str = "2023-12-31"
):
    """
    Generate a 6-panel comprehensive dashboard showing all rolling OU parameters,
    half-life, equilibrium standard deviation, and standardized spread.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=300)
    split_dt = pd.Timestamp(train_split_date)
    valid_mask = df["Is_Valid"]
    dates = df.index[valid_mask]
    
    # 1. Mean-Reversion Speed Theta
    ax = axes[0, 0]
    ax.plot(dates, df.loc[valid_mask, "Theta"], color="#1f77b4", linewidth=1.3)
    ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title(r"$\theta$ (Mean-Reversion Speed, Annualised)", fontweight="bold")
    ax.set_ylabel(r"$\theta$ (yr$^{-1}$)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Long-Term Mean Mu vs Spread
    ax = axes[0, 1]
    ax.plot(df.index, df["Spread_X"], color="#7f7f7f", alpha=0.5, linewidth=1.0, label="Spread $X_t$")
    ax.plot(dates, df.loc[valid_mask, "Mu"], color="#2ca02c", linewidth=1.4, label=r"Rolling $\mu$")
    ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title(r"$\mu$ (Long-Run Equilibrium Mean) vs Spread $X_t$", fontweight="bold")
    ax.set_ylabel(r"$\mu$ / $X_t$", fontweight="bold")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 3. Diffusion Volatility Sigma
    ax = axes[1, 0]
    ax.plot(dates, df.loc[valid_mask, "Sigma"], color="#9467bd", linewidth=1.3)
    ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title(r"$\sigma$ (Diffusion Volatility, Annualised)", fontweight="bold")
    ax.set_ylabel(r"$\sigma$", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 4. Half-Life (Trading Days)
    ax = axes[1, 1]
    ax.plot(dates, df.loc[valid_mask, "Half_Life_Days"], color="#ff7f0e", linewidth=1.3, label="Half-Life $t_{1/2}$")
    ax.axhline(5.0, color="#2ca02c", linestyle=":", linewidth=1.2, label="Lower Bound (5 days)")
    ax.axhline(60.0, color="#d62728", linestyle=":", linewidth=1.2, label="Upper Bound (60 days)")
    ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title(r"Half-Life $t_{1/2}$ (Trading Days)", fontweight="bold")
    ax.set_ylabel("Days", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 5. Equilibrium Standard Deviation Sigma_eq
    ax = axes[2, 0]
    ax.plot(dates, df.loc[valid_mask, "Sigma_Eq"], color="#8c564b", linewidth=1.3)
    ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title(r"$\sigma_{eq} = \sigma / \sqrt{2\theta}$ (Equilibrium Std Dev)", fontweight="bold")
    ax.set_ylabel(r"$\sigma_{eq}$", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 6. Standardised Spread s_t
    ax = axes[2, 1]
    ax.plot(dates, df.loc[valid_mask, "Z_Score"], color="#1f77b4", linewidth=1.2)
    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8)
    ax.axhline(2.0, color="#d95f02", linestyle="--", linewidth=1.0)
    ax.axhline(-2.0, color="#1b9e77", linestyle="--", linewidth=1.0)
    ax.axhline(3.5, color="#e7298a", linestyle="-.", linewidth=1.0)
    ax.axhline(-3.5, color="#e7298a", linestyle="-.", linewidth=1.0)
    ax.axvline(split_dt, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title("Standardised Spread $s_t$ (Z-Score)", fontweight="bold")
    ax.set_ylabel("$s_t$", fontweight="bold")
    ax.set_ylim(-4.5, 4.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    for row in axes:
        for ax in row:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
    
    fig.suptitle(f"Ornstein-Uhlenbeck Rolling Estimation Dashboard — {pair_name}\n"
                 f"(Window: 120 Days, $\\Delta t = 1/252$ yr, Split: {train_split_date})",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
