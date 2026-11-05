"""
Pair Selection and Cointegration Analysis Module for OU Pairs Trading.

Course: MAC F244 — Stochastic Calculus and Applications to Finance
Phase 2: Pair Selection and Cointegration

Implements:
1. Pearson correlation screening of training log-prices.
2. OLS hedge ratio estimation (beta) and spread construction: X_t = ln(P_A) - beta * ln(P_B).
3. Engle-Granger cointegration testing with MacKinnon p-values and ADF residual tests at alpha = 0.05.
4. Preliminary AR(1) discretisation for Ornstein-Uhlenbeck parameters and half-life (5-60 trading days).
5. Comprehensive reporting and publication-grade visualizations.
"""

import os
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_correlation_matrix(df_log_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Pearson correlation matrix of log-prices across all assets.

    Parameters
    ----------
    df_log_prices : pd.DataFrame
        Training period log-prices with DatetimeIndex and asset ticker columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    return df_log_prices.corr(method="pearson")


def estimate_hedge_ratio(
    s_a: pd.Series,
    s_b: pd.Series,
) -> Dict[str, Any]:
    """
    Estimate the hedge ratio beta by regressing log(P_A) on log(P_B) with OLS.

    Model: ln(P_A,t) = alpha + beta * ln(P_B,t) + epsilon_t
    Spread: X_t = ln(P_A,t) - beta * ln(P_B,t)

    Parameters
    ----------
    s_a : pd.Series
        Dependent asset log-price series ln(P_A).
    s_b : pd.Series
        Independent asset log-price series ln(P_B).

    Returns
    -------
    Dict[str, Any]
        Dictionary with alpha, beta, r_squared, t_stat, residuals, and spread.
    """
    X = sm.add_constant(s_b)
    model = sm.OLS(s_a, X).fit()

    alpha = float(model.params["const"])
    beta = float(model.params[s_b.name])
    r_squared = float(model.rsquared)
    t_stat_beta = float(model.tvalues[s_b.name])
    p_val_beta = float(model.pvalues[s_b.name])

    # Residuals: epsilon_t = ln(P_A,t) - alpha - beta * ln(P_B,t)
    residuals = model.resid

    # Spread without constant: X_t = ln(P_A,t) - beta * ln(P_B,t)
    spread = s_a - beta * s_b

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "t_stat_beta": t_stat_beta,
        "p_val_beta": p_val_beta,
        "residuals": residuals,
        "spread": spread,
        "model": model,
    }


def test_engle_granger_cointegration(
    s_a: pd.Series,
    s_b: pd.Series,
    residuals: Optional[pd.Series] = None,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform the Engle-Granger two-step cointegration test.

    Tests the null hypothesis that no cointegrating relationship exists
    (i.e., the regression residuals contain a unit root).

    Parameters
    ----------
    s_a : pd.Series
        Log-price series for Stock A.
    s_b : pd.Series
        Log-price series for Stock B.
    residuals : Optional[pd.Series]
        OLS residuals if already calculated.
    significance_level : float
        Significance threshold (default 0.05).

    Returns
    -------
    Dict[str, Any]
        Cointegration test statistics, p-values, critical values, and decision.
    """
    # 1. Statsmodels Engle-Granger coint test (applies MacKinnon cointegration critical values)
    coint_stat, coint_pval, crit_values = coint(s_a, s_b)

    # 2. Also run Augmented Dickey-Fuller test on the OLS regression residuals
    if residuals is None:
        X = sm.add_constant(s_b)
        model = sm.OLS(s_a, X).fit()
        residuals = model.resid

    adf_result = adfuller(residuals, autolag="AIC")
    adf_stat = float(adf_result[0])
    adf_pval = float(adf_result[1])
    used_lag = int(adf_result[2])
    n_obs = int(adf_result[3])
    adf_crits = adf_result[4]

    # Cointegration decision: reject H0 at the specified significance level
    is_cointegrated = (coint_pval < significance_level) or (adf_pval < significance_level)

    return {
        "coint_stat": float(coint_stat),
        "coint_pvalue": float(coint_pval),
        "coint_crit_1pct": float(crit_values[0]),
        "coint_crit_5pct": float(crit_values[1]),
        "coint_crit_10pct": float(crit_values[2]),
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pval,
        "adf_lags": used_lag,
        "adf_crit_5pct": float(adf_crits["5%"]),
        "is_cointegrated": is_cointegrated,
    }


def estimate_preliminary_ou(
    spread: pd.Series,
    dt: float = 1.0 / 252.0,
) -> Dict[str, Any]:
    """
    Fit a preliminary AR(1) discretisation to estimate OU speed theta and half-life.

    Exact discretisation:
        X_{t+Delta} = a + b * X_t + epsilon_t
        b = exp(-theta * Delta)
        theta = -ln(b) / Delta
        half_life (years) = ln(2) / theta
        half_life (trading days) = half_life * 252 = ln(2) / (-ln(b))

    Parameters
    ----------
    spread : pd.Series
        Spread series X_t.
    dt : float
        Annualized time step Delta (1/252 for daily trading data).

    Returns
    -------
    Dict[str, Any]
        AR(1) parameter b, mean-reversion speed theta, half-life, and validity flag.
    """
    x_t = spread.iloc[:-1].values
    x_tp1 = spread.iloc[1:].values

    X_mat = sm.add_constant(x_t)
    reg_ar = sm.OLS(x_tp1, X_mat).fit()

    a_const = float(reg_ar.params[0])
    b_coef = float(reg_ar.params[1])
    r2_ar = float(reg_ar.rsquared)
    resid_var = float(np.var(reg_ar.resid, ddof=2))

    # Numerical safeguard: Check 0 < b < 1 for valid mean-reverting OU process
    if not (0.0 < b_coef < 1.0):
        # b >= 1 represents unit root or explosive behavior; b <= 0 represents oscillatory divergence
        reason = "Unit root or explosive (b >= 1)" if b_coef >= 1.0 else "Oscillatory divergence (b <= 0)"
        return {
            "ar1_a": a_const,
            "ar1_b": b_coef,
            "ar1_r2": r2_ar,
            "theta_annualized": np.nan,
            "half_life_days": np.nan,
            "mu_equilibrium": np.nan,
            "sigma_annualized": np.nan,
            "is_valid_ou": False,
            "invalidation_reason": reason,
        }

    # Recover annualised OU parameters
    theta_annualized = -np.log(b_coef) / dt
    mu_eq = a_const / (1.0 - b_coef)

    # Annualised volatility sigma
    denom = 1.0 - (b_coef ** 2)
    if denom > 0:
        sigma_annualized = np.sqrt(2.0 * theta_annualized * resid_var / denom)
    else:
        sigma_annualized = np.nan

    # Half-life in trading days
    # Half-life in years = ln(2) / theta; in trading days = (ln(2) / theta) * 252 = ln(2) / -ln(b)
    half_life_days = np.log(2.0) / (-np.log(b_coef))

    # Half-life filter: 5 to 60 trading days
    meets_half_life_criteria = (5.0 <= half_life_days <= 60.0)

    return {
        "ar1_a": a_const,
        "ar1_b": b_coef,
        "ar1_r2": r2_ar,
        "theta_annualized": float(theta_annualized),
        "half_life_days": float(half_life_days),
        "mu_equilibrium": float(mu_eq),
        "sigma_annualized": float(sigma_annualized),
        "is_valid_ou": True,
        "meets_half_life_criteria": meets_half_life_criteria,
        "invalidation_reason": None,
    }


def analyze_all_pairs(
    df_log_prices: pd.DataFrame,
    ticker_names: Optional[Dict[str, str]] = None,
    significance_level: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all 15 unique pairs across correlation, OLS hedge ratio,
    Engle-Granger cointegration, and OU half-life.

    Parameters
    ----------
    df_log_prices : pd.DataFrame
        Training-period log-prices (2019-2023).
    ticker_names : Optional[Dict[str, str]]
        Mapping of tickers to full asset names.
    significance_level : float
        Cointegration p-value threshold (0.05).
    min_half_life : float
        Minimum acceptable half-life in trading days (5.0).
    max_half_life : float
        Maximum acceptable half-life in trading days (60.0).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - all_pairs_df: Analysis table for all 15 candidate pairs.
        - cointegrated_df: Subset of pairs passing cointegration test (p < 0.05).
        - selected_df: Final selected pairs passing both cointegration and half-life criteria.
    """
    tickers = list(df_log_prices.columns)
    pair_combos = list(combinations(tickers, 2))

    all_records: List[Dict[str, Any]] = []

    for stock_a, stock_b in pair_combos:
        name_a = ticker_names.get(stock_a, stock_a) if ticker_names else stock_a
        name_b = ticker_names.get(stock_b, stock_b) if ticker_names else stock_b
        pair_label = f"{stock_a} - {stock_b}"
        pair_display = f"{name_a} & {name_b}"

        s_a = df_log_prices[stock_a]
        s_b = df_log_prices[stock_b]

        # 1. Pearson Correlation
        corr = float(s_a.corr(s_b))

        # 2. OLS Hedge Ratio Estimation
        ols_res = estimate_hedge_ratio(s_a, s_b)
        alpha = ols_res["alpha"]
        beta = ols_res["beta"]
        r2 = ols_res["r_squared"]
        residuals = ols_res["residuals"]
        spread = ols_res["spread"]

        # 3. Engle-Granger Cointegration Testing
        coint_res = test_engle_granger_cointegration(
            s_a, s_b, residuals=residuals, significance_level=significance_level
        )
        coint_stat = coint_res["coint_stat"]
        coint_pval = coint_res["coint_pvalue"]
        adf_stat = coint_res["adf_stat"]
        adf_pval = coint_res["adf_pvalue"]
        is_coint = coint_res["is_cointegrated"]

        # 4. Preliminary OU Parameter & Half-Life Estimation
        ou_res = estimate_preliminary_ou(spread, dt=1.0 / 252.0)
        ar1_b = ou_res["ar1_b"]
        theta = ou_res["theta_annualized"]
        half_life = ou_res["half_life_days"]
        mu_eq = ou_res["mu_equilibrium"]
        sigma_ou = ou_res["sigma_annualized"]
        is_valid_ou = ou_res["is_valid_ou"]
        meets_hl = ou_res.get("meets_half_life_criteria", False)

        # 5. Screening Decision and Selection Status
        if not is_coint:
            status = "REJECTED (No Cointegration)"
            reason = f"Cointegration p-value ({min(coint_pval, adf_pval):.4f}) >= {significance_level}"
        elif not is_valid_ou:
            status = "REJECTED (Non-Mean-Reverting)"
            reason = ou_res["invalidation_reason"]
        elif not meets_hl:
            status = "REJECTED (Half-Life Out of Bounds)"
            reason = f"Half-life ({half_life:.1f} d) not in [{min_half_life}, {max_half_life}] days"
        else:
            status = "SELECTED"
            reason = f"Cointegrated (p={min(coint_pval, adf_pval):.4f} < 0.05) & Half-Life ({half_life:.1f} d) within [{min_half_life}, {max_half_life}] days"

        all_records.append({
            "Pair": pair_label,
            "Pair Names": pair_display,
            "Stock A (Y)": stock_a,
            "Stock B (X)": stock_b,
            "Correlation": round(corr, 4),
            "Hedge Ratio (Beta)": round(beta, 4),
            "Alpha (Intercept)": round(alpha, 4),
            "OLS R-Squared": round(r2, 4),
            "EG Coint Stat": round(coint_stat, 4),
            "EG Coint p-value": round(coint_pval, 4),
            "Residual ADF Stat": round(adf_stat, 4),
            "Residual ADF p-value": round(adf_pval, 4),
            "Cointegrated (5%)": "YES" if is_coint else "NO",
            "AR(1) b": round(ar1_b, 6),
            "Theta (Speed)": round(theta, 4) if not np.isnan(theta) else np.nan,
            "Half-Life (Days)": round(half_life, 2) if not np.isnan(half_life) else np.nan,
            "Equilibrium Mean (Mu)": round(mu_eq, 4) if not np.isnan(mu_eq) else np.nan,
            "OU Volatility (Sigma)": round(sigma_ou, 4) if not np.isnan(sigma_ou) else np.nan,
            "Selection Status": status,
            "Decision Reason": reason,
        })

    all_pairs_df = pd.DataFrame(all_records)

    # Cointegrated pairs subset
    cointegrated_df = all_pairs_df[all_pairs_df["Cointegrated (5%)"] == "YES"].copy()

    # Final selected pairs subset
    selected_df = all_pairs_df[all_pairs_df["Selection Status"] == "SELECTED"].copy()

    return all_pairs_df, cointegrated_df, selected_df


def plot_correlation_matrix(
    df_log_prices: pd.DataFrame,
    save_path: str,
    ticker_names: Optional[Dict[str, str]] = None,
) -> str:
    """Generate and save correlation matrix heatmap."""
    corr = calculate_correlation_matrix(df_log_prices)
    labels = [ticker_names.get(c, c) if ticker_names else c for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    cax = ax.imshow(corr.values, cmap="YlGnBu", interpolation="nearest", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")

    # Annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            color = "white" if val > 0.75 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_title("Nifty Bank Constituents: Log-Price Correlation Matrix (Training: 2019 - 2023)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_correlation_vs_cointegration(
    all_pairs_df: pd.DataFrame,
    save_path: str,
) -> str:
    """
    Comparison scatter plot illustrating why high correlation does NOT imply cointegration.
    """
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # Use the minimum of Engle-Granger and Residual ADF p-value as effective cointegration p-value
    p_vals = np.minimum(all_pairs_df["EG Coint p-value"], all_pairs_df["Residual ADF p-value"])
    corrs = all_pairs_df["Correlation"]

    colors = ["#2ca02c" if status == "SELECTED" else ("#1f77b4" if "Cointegrated" in status or is_coint == "YES" else "#d62728")
              for status, is_coint in zip(all_pairs_df["Selection Status"], all_pairs_df["Cointegrated (5%)"])]

    scatter = ax.scatter(corrs, p_vals, c=colors, s=120, edgecolors="black", linewidth=1.2, zorder=3)

    # 5% significance threshold line
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="5% Cointegration Significance Level (p = 0.05)")

    # Annotate selected and notable pairs
    for _, row in all_pairs_df.iterrows():
        p_val = min(row["EG Coint p-value"], row["Residual ADF p-value"])
        pair_short = row["Pair"].replace(".NS", "")
        if row["Selection Status"] == "SELECTED" or p_val < 0.05 or row["Correlation"] > 0.90:
            ax.annotate(pair_short, (row["Correlation"], p_val),
                        xytext=(6, 5), textcoords="offset points", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="#cccccc"))

    ax.set_title("Correlation vs. Cointegration p-value Across All 15 Candidate Pairs", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Pearson Correlation (Training Log-Prices)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cointegration Test p-value", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, 0.90)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Selected Pair (Cointegrated + Valid Half-Life)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Rejected Pair (Not Cointegrated, p >= 0.05)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='5% Critical Boundary (p = 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_cointegration_pvalues(
    all_pairs_df: pd.DataFrame,
    save_path: str,
) -> str:
    """Generate bar chart of cointegration p-values across all 15 pairs."""
    sorted_df = all_pairs_df.sort_values(by="EG Coint p-value", ascending=True).reset_index(drop=True)
    pair_labels = [p.replace(".NS", "") for p in sorted_df["Pair"]]
    p_vals = sorted_df["EG Coint p-value"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bar_colors = ["#2ca02c" if p < 0.05 else "#4a7bb0" for p in p_vals]

    bars = ax.bar(range(len(pair_labels)), p_vals, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.6)

    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="5% Significance Threshold (p = 0.05)")

    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=40, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Engle-Granger Cointegration p-value", fontsize=11, fontweight="bold")
    ax.set_title("Engle-Granger Cointegration Test p-values Across All 15 Pairs (Training: 2019 - 2023)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)

    # Annotate bars
    for bar, p in zip(bars, p_vals):
        y_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_val + 0.015, f"{p:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(p_vals) * 1.15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_selected_pair_spreads(
    selected_df: pd.DataFrame,
    df_log_prices: pd.DataFrame,
    save_path: str,
    ticker_names: Optional[Dict[str, str]] = None,
) -> str:
    """
    Plot the historical log-price spread X_t = ln(P_A) - beta * ln(P_B)
    for all selected pairs over the training period.
    """
    n_pairs = len(selected_df)
    if n_pairs == 0:
        return ""

    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 5 * n_pairs), dpi=300, squeeze=False)

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        ax = axes[idx, 0]
        stock_a = row["Stock A (Y)"]
        stock_b = row["Stock B (X)"]
        beta = row["Hedge Ratio (Beta)"]
        name_a = ticker_names.get(stock_a, stock_a) if ticker_names else stock_a
        name_b = ticker_names.get(stock_b, stock_b) if ticker_names else stock_b

        spread = df_log_prices[stock_a] - beta * df_log_prices[stock_b]
        mu = spread.mean()
        std = spread.std()

        ax.plot(spread.index, spread.values, label=f"Spread: {stock_a} - {beta:.4f} * {stock_b}", color="#1f77b4", linewidth=1.5)
        ax.axhline(mu, color="black", linestyle="-", linewidth=1.2, label=f"Mean (μ = {mu:.4f})")
        ax.axhline(mu + 1.96 * std, color="red", linestyle="--", linewidth=1.0, alpha=0.8, label=f"+1.96σ ({mu + 1.96 * std:.4f})")
        ax.axhline(mu - 1.96 * std, color="green", linestyle="--", linewidth=1.0, alpha=0.8, label=f"-1.96σ ({mu - 1.96 * std:.4f})")

        title = (
            f"Selected Pair Spread: {name_a} vs. {name_b}\n"
            f"Hedge Ratio β = {beta:.4f} | EG p-val = {row['EG Coint p-value']:.4f} | "
            f"θ = {row['Theta (Speed)']:.2f} yr⁻¹ | Half-Life = {row['Half-Life (Days)']:.1f} trading days"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10, fontweight="bold")
        ax.set_ylabel("Spread X_t = ln(P_A) - β ln(P_B)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path
