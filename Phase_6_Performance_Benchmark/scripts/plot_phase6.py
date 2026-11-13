"""High-Resolution Publication-Quality Plotting Suite for Phase 6.

Generates 7 publication-ready figures:
1. `benchmark_equity_curves.png`: Out-of-sample cumulative net equity across strategies and market benchmarks.
2. `benchmark_drawdowns.png`: Comparative underwater drawdown profiles (%).
3. `fpt_theoretical_surface.png`: Continuous-time OU first-passage-time return surface and holding time contours.
4. `threshold_sensitivity_heatmap.png`: 2D parameter heatmaps of Net Return, Sharpe Ratio, Win Rate, and Drawdown.
5. `transaction_cost_impact.png`: Strategy profitability and Sharpe ratio decay across transaction cost regimes.
6. `stop_loss_sensitivity.png`: Risk-return tradeoff across stop-loss barrier levels.
7. `performance_comparison_bar.png`: Multi-metric bar chart comparing strategy variants and market benchmarks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_benchmark_equity_curves(
    equity_df: pd.DataFrame,
    save_path: Path,
):
    """Plot comparative cumulative net equity curves against market benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    dates = pd.to_datetime(equity_df["Date"])

    # Line styling palette
    styles = {
        "Fixed Benchmark (2.0 / 0.5)__PORTFOLIO__Net_Equity": ("#1f77b4", "-", 2.2, "Fixed Benchmark (2.0 / 0.5) Portfolio"),
        "Analytical Optimal (FPT)__PORTFOLIO__Net_Equity": ("#2ca02c", "-", 2.5, "Analytical Optimal (FPT) Portfolio"),
        "In-Sample Calibrated (1.75 / 0.25)__PORTFOLIO__Net_Equity": ("#ff7f0e", "--", 2.0, "In-Sample Calibrated Portfolio"),
        "Fixed Benchmark (2.0 / 0.5)__HDFCBANK.NS - KOTAKBANK.NS__Net_Equity": ("#9467bd", ":", 1.5, "Fixed: HDFC - Kotak"),
        "Analytical Optimal (FPT)__HDFCBANK.NS - KOTAKBANK.NS__Net_Equity": ("#d62728", "-.", 2.0, "Optimal FPT: HDFC - Kotak"),
    }

    for col, (color, ls, lw, label) in styles.items():
        if col in equity_df.columns:
            ax.plot(dates, equity_df[col], color=color, linestyle=ls, linewidth=lw, label=label)

    # Reference initial wealth line
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Initial Capital ($1.00)")

    ax.set_title("Out-of-Sample Net Cumulative Equity Comparison (2024–2026, 680 Trading Days)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Net Portfolio Equity (Base = 1.0)", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=25)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_benchmark_drawdowns(
    equity_df: pd.DataFrame,
    save_path: Path,
):
    """Plot comparative underwater drawdown profiles (%) for strategy variants."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    dates = pd.to_datetime(equity_df["Date"])

    strat_cols = [
        ("Fixed Benchmark (2.0 / 0.5)__PORTFOLIO__Net_Equity", "#1f77b4", "Fixed Benchmark Portfolio"),
        ("Analytical Optimal (FPT)__PORTFOLIO__Net_Equity", "#2ca02c", "Analytical Optimal Portfolio"),
        ("In-Sample Calibrated (1.75 / 0.25)__PORTFOLIO__Net_Equity", "#ff7f0e", "In-Sample Calibrated Portfolio"),
    ]

    for col, color, label in strat_cols:
        if col in equity_df.columns:
            eq = equity_df[col].values
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / peak * 100.0  # %
            ax.plot(dates, -dd, color=color, linewidth=1.8, label=label)
            ax.fill_between(dates, -dd, 0, color=color, alpha=0.1)

    ax.set_title("Out-of-Sample Underwater Drawdown Profiles (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=25)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_fpt_theoretical_surface(
    se_grid: np.ndarray,
    sx_grid: np.ndarray,
    ret_surface: np.ndarray,
    duration_surface: np.ndarray,
    optimal_point: Tuple[float, float],
    heuristic_point: Tuple[float, float],
    save_path: Path,
):
    """Plot the continuous-time OU first-passage-time return rate and holding duration contours."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

    # Panel 1: Return rate contour
    cp1 = ax1.contourf(se_grid, sx_grid, ret_surface, levels=25, cmap="viridis")
    cbar1 = fig.colorbar(cp1, ax=ax1)
    cbar1.set_label("Theoretical Annual Return (% p.a.)", fontweight="bold")

    ax1.plot(optimal_point[0], optimal_point[1], "r*", markersize=14, label=f"Optimal FPT ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})")
    ax1.plot(heuristic_point[0], heuristic_point[1], "wo", markersize=8, markeredgecolor="black", label=f"Heuristic ({heuristic_point[0]:.1f}, {heuristic_point[1]:.1f})")

    ax1.set_title("(a) Theoretical Rate of Return Surface $\\rho(s_e, s_x)$", fontweight="bold")
    ax1.set_xlabel("Entry Threshold $s_e$ (Dimensionless)", fontweight="bold")
    ax1.set_ylabel("Exit Threshold $s_x$ (Dimensionless)", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white")

    # Panel 2: Trade duration contour
    cp2 = ax2.contourf(se_grid, sx_grid, duration_surface, levels=25, cmap="magma")
    cbar2 = fig.colorbar(cp2, ax=ax2)
    cbar2.set_label("Expected Holding Period (Trading Days)", fontweight="bold")

    ax2.plot(optimal_point[0], optimal_point[1], "w*", markersize=14, markeredgecolor="black")
    ax2.plot(heuristic_point[0], heuristic_point[1], "wo", markersize=8, markeredgecolor="black")

    ax2.set_title("(b) Expected Trade Duration $E[T_{trade}]$ via Dynkin's ODE", fontweight="bold")
    ax2.set_xlabel("Entry Threshold $s_e$ (Dimensionless)", fontweight="bold")
    ax2.set_ylabel("Exit Threshold $s_x$ (Dimensionless)", fontweight="bold")

    plt.suptitle("Stochastic Calculus of OU Thresholds: Continuous-Time FPT Solution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_threshold_sensitivity_heatmap(
    grid_df: pd.DataFrame,
    pair_name: str,
    save_path: Path,
):
    """Plot 2x2 heatmap of Net Return, Sharpe Ratio, Win Rate, and Max Drawdown."""
    p_df = grid_df[grid_df["Pair"] == pair_name].copy()
    if p_df.empty:
        p_df = grid_df.copy()

    ret_pivot = p_df.pivot(index="Exit_Threshold", columns="Entry_Threshold", values="Cumulative_Net_Return") * 100.0
    sharpe_pivot = p_df.pivot(index="Exit_Threshold", columns="Entry_Threshold", values="Sharpe_Ratio_rf_zero")
    win_pivot = p_df.pivot(index="Exit_Threshold", columns="Entry_Threshold", values="Win_Rate") * 100.0
    dd_pivot = p_df.pivot(index="Exit_Threshold", columns="Entry_Threshold", values="Max_Drawdown_Net") * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    sns.heatmap(ret_pivot, ax=axes[0, 0], cmap="RdYlGn", annot=True, fmt=".1f", cbar_kws={"label": "Net Return (%)"})
    axes[0, 0].set_title("(a) Cumulative Net Return (%)", fontweight="bold")
    axes[0, 0].set_xlabel("Entry Threshold $s_e$")
    axes[0, 0].set_ylabel("Exit Threshold $s_x$")

    sns.heatmap(sharpe_pivot, ax=axes[0, 1], cmap="coolwarm", annot=True, fmt=".2f", center=0.0, cbar_kws={"label": "Sharpe Ratio ($r_f=0$)"})
    axes[0, 1].set_title("(b) Annualized Sharpe Ratio", fontweight="bold")
    axes[0, 1].set_xlabel("Entry Threshold $s_e$")
    axes[0, 1].set_ylabel("Exit Threshold $s_x$")

    sns.heatmap(win_pivot, ax=axes[1, 0], cmap="Blues", annot=True, fmt=".1f", cbar_kws={"label": "Win Rate (%)"})
    axes[1, 0].set_title("(c) Strategy Win Rate (%)", fontweight="bold")
    axes[1, 0].set_xlabel("Entry Threshold $s_e$")
    axes[1, 0].set_ylabel("Exit Threshold $s_x$")

    sns.heatmap(dd_pivot, ax=axes[1, 1], cmap="Reds_r", annot=True, fmt=".1f", cbar_kws={"label": "Max Drawdown (%)"})
    axes[1, 1].set_title("(d) Maximum Drawdown (%)", fontweight="bold")
    axes[1, 1].set_xlabel("Entry Threshold $s_e$")
    axes[1, 1].set_ylabel("Exit Threshold $s_x$")

    plt.suptitle(f"Threshold Sensitivity Grid Analysis: {pair_name}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_transaction_cost_impact(
    cost_df: pd.DataFrame,
    save_path: Path,
):
    """Plot Net Return and Sharpe Ratio degradation across transaction cost regimes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    palette = {"HDFCBANK.NS - KOTAKBANK.NS": "#1f77b4", "HDFCBANK.NS - SBIN.NS": "#2ca02c",
               "KOTAKBANK.NS - SBIN.NS": "#d62728", "COMBINED PORTFOLIO": "#9467bd"}

    for pair, grp in cost_df.groupby("Pair"):
        color = palette.get(pair, "#333333")
        lw = 2.5 if pair == "COMBINED PORTFOLIO" else 1.8
        ls = "-" if pair == "COMBINED PORTFOLIO" else "--"
        label = "Combined Portfolio" if pair == "COMBINED PORTFOLIO" else pair.replace(".NS", "")
        ax1.plot(grp["Cost_Bps_Per_Leg"], grp["Cumulative_Net_Return"] * 100.0, color=color, linewidth=lw, linestyle=ls, marker="o", label=label)
        ax2.plot(grp["Cost_Bps_Per_Leg"], grp["Sharpe_Ratio_rf_zero"], color=color, linewidth=lw, linestyle=ls, marker="s", label=label)

    ax1.axhline(0.0, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.set_title("(a) Cumulative Net Return vs. Friction", fontweight="bold")
    ax1.set_xlabel("Transaction Cost per Leg (Basis Points)", fontweight="bold")
    ax1.set_ylabel("Net Cumulative Return (%)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_title("(b) Annualized Sharpe Ratio ($r_f=0$) vs. Friction", fontweight="bold")
    ax2.set_xlabel("Transaction Cost per Leg (Basis Points)", fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    plt.suptitle("Strategy Performance Sensitivity to Brokerage and Execution Friction", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_stop_loss_sensitivity(
    stop_df: pd.DataFrame,
    save_path: Path,
):
    """Plot risk-return tradeoffs across stop-loss thresholds."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    port_df = stop_df[stop_df["Pair"] == "COMBINED PORTFOLIO"].copy()

    x_labels = [str(x) for x in port_df["Stop_Loss_Z"].values]
    x_pos = np.arange(len(x_labels))

    ax1.bar(x_pos - 0.2, port_df["Cumulative_Net_Return"] * 100.0, width=0.4, color="#1f77b4", label="Cumulative Net Return (%)")
    ax1.set_ylabel("Net Return (%)", color="#1f77b4", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Stop-Loss Threshold $s_{stop}$ (Dimensionless Z-Score)", fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)

    ax2 = ax1.twinx()
    ax2.plot(x_pos + 0.2, port_df["Max_Drawdown_Net"] * 100.0, color="#d62728", marker="o", linewidth=2.2, label="Max Drawdown (%)")
    ax2.set_ylabel("Max Drawdown (%)", color="#d62728", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Stop-Loss Barrier Analysis: Capital Preservation vs. Premature Stop-Outs", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_performance_comparison_bar(
    metrics_df: pd.DataFrame,
    save_path: Path,
):
    """Generate comparative bar chart of key risk-adjusted metrics across strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Filter to main strategy comparison entries
    strats = ["Fixed Benchmark (2.0 / 0.5)", "Analytical Optimal (FPT)", "In-Sample Calibrated (1.75 / 0.25)"]
    sub_df = metrics_df[(metrics_df["Strategy"].isin(strats)) & (metrics_df["Pair"] == "COMBINED PORTFOLIO")].copy()

    labels = ["Fixed Benchmark", "Analytical Optimal", "In-Sample Calibrated"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # (a) Net Return
    axes[0, 0].bar(labels, sub_df["Cumulative_Net_Return"] * 100.0, color=colors, width=0.55)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("(a) Cumulative Net Return (%)", fontweight="bold")
    axes[0, 0].set_ylabel("% Return")

    # (b) Sharpe Ratio
    axes[0, 1].bar(labels, sub_df["Sharpe_Ratio_rf_zero"], color=colors, width=0.55)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("(b) Annualized Sharpe Ratio ($r_f=0$)", fontweight="bold")
    axes[0, 1].set_ylabel("Sharpe Ratio")

    # (c) Max Drawdown
    axes[1, 0].bar(labels, sub_df["Max_Drawdown_Net"] * 100.0, color=colors, width=0.55)
    axes[1, 0].set_title("(c) Maximum Drawdown (%)", fontweight="bold")
    axes[1, 0].set_ylabel("% Drawdown")

    # (d) Annualized Turnover
    axes[1, 1].bar(labels, sub_df["Annualised_Turnover"], color=colors, width=0.55)
    axes[1, 1].set_title("(d) Annualized Portfolio Turnover (Turns/Year)", fontweight="bold")
    axes[1, 1].set_ylabel("Turnover (x)")

    plt.suptitle("Out-of-Sample Performance Comparison: Fixed vs. Analytical Optimal Strategies", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

