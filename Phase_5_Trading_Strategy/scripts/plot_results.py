"""Plotting utilities for Phase 5 OU Trading Strategy and Backtest.

Generates high-resolution publication-quality figures:
1. Equity curves (Single pairs and combined portfolio).
2. Drawdown curves.
3. Standardised spread with entry/exit/stop-loss thresholds and trade markers.
4. Multi-panel trade visualization across all pairs.
5. Risk-adjusted performance bar chart comparison.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def set_plot_style():
    """Configure matplotlib style for clean financial charts."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5


def plot_equity_curves(
    pair_daily_dict: Dict[str, pd.DataFrame],
    portfolio_daily_df: pd.DataFrame,
    output_path: str,
):
    """Plot net cumulative equity curves for all pairs and the combined portfolio."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Color palette
    colors = {
        "HDFCBANK.NS - KOTAKBANK.NS": "#1f77b4",  # Blue
        "HDFCBANK.NS - SBIN.NS": "#2ca02c",       # Green
        "KOTAKBANK.NS - SBIN.NS": "#ff7f0e",      # Orange
    }

    dates = pd.to_datetime(portfolio_daily_df["Date"])

    # Plot individual pairs
    for pair, df in pair_daily_dict.items():
        c = colors.get(pair, "#7f7f7f")
        final_ret = (df["Net_Equity"].iloc[-1] - 1.0) * 100.0
        ax.plot(
            dates,
            df["Net_Equity"],
            label=f"{pair} (Net: {final_ret:+.2f}%)",
            color=c,
            linewidth=1.5,
            alpha=0.75,
        )

    # Plot combined portfolio
    port_final_ret = (portfolio_daily_df["Portfolio_Net_Equity"].iloc[-1] - 1.0) * 100.0
    ax.plot(
        dates,
        portfolio_daily_df["Portfolio_Net_Equity"],
        label=f"Combined Portfolio (Equal Weight, Net: {port_final_ret:+.2f}%)",
        color="#d62728",  # Bold Red
        linewidth=2.5,
    )

    # Baseline
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0, alpha=0.8, label="Initial Capital (1.0)")

    ax.set_title("Phase 5: Out-of-Sample Cumulative Net Equity Curves (2024 - 2026)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trading Date", fontsize=11, labelpad=8)
    ax.set_ylabel("Portfolio Net Equity (Base = 1.0)", fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_drawdowns(
    pair_daily_dict: Dict[str, pd.DataFrame],
    portfolio_daily_df: pd.DataFrame,
    output_path: str,
):
    """Plot underwater drawdown profiles for each pair and the combined portfolio."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    dates = pd.to_datetime(portfolio_daily_df["Date"])

    colors = {
        "HDFCBANK.NS - KOTAKBANK.NS": "#1f77b4",
        "HDFCBANK.NS - SBIN.NS": "#2ca02c",
        "KOTAKBANK.NS - SBIN.NS": "#ff7f0e",
    }

    for pair, df in pair_daily_dict.items():
        c = colors.get(pair, "#7f7f7f")
        max_dd = df["Net_Drawdown"].max() * 100.0
        ax.plot(
            dates,
            -df["Net_Drawdown"] * 100.0,
            label=f"{pair} (Max DD: -{max_dd:.2f}%)",
            color=c,
            linewidth=1.2,
            alpha=0.6,
        )

    # Combined portfolio
    port_max_dd = portfolio_daily_df["Portfolio_Net_Drawdown"].max() * 100.0
    ax.fill_between(
        dates,
        -portfolio_daily_df["Portfolio_Net_Drawdown"] * 100.0,
        0,
        color="#d62728",
        alpha=0.2,
    )
    ax.plot(
        dates,
        -portfolio_daily_df["Portfolio_Net_Drawdown"] * 100.0,
        label=f"Combined Portfolio (Max DD: -{port_max_dd:.2f}%)",
        color="#d62728",
        linewidth=2.0,
    )

    ax.set_title("Phase 5: Out-of-Sample Portfolio Underwater Drawdown (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trading Date", fontsize=11, labelpad=8)
    ax.set_ylabel("Drawdown (%)", fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pair_spread_and_trades(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    pair_name: str,
    output_path: str,
    entry_thresh: float = 2.0,
    exit_thresh: float = 0.5,
    stop_loss_thresh: float = 3.5,
):
    """Plot standardized spread s_t with threshold bands and exact trade entry/exit markers."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.0]})

    dates = pd.to_datetime(daily_df["Date"])
    z = daily_df["Z_Score"]

    # --- Top Panel: Standardised Spread and Trades ---
    ax1.plot(dates, z, color="#2b5c8f", linewidth=1.2, label=r"Standardised Spread $s_t = (X_t - \mu_t) / \sigma_{eq, t}$")

    # Threshold horizontal lines
    ax1.axhline(+stop_loss_thresh, color="#b2182b", linestyle="--", linewidth=1.2, label=f"Stop-Loss (+{stop_loss_thresh})")
    ax1.axhline(+entry_thresh, color="#d6604d", linestyle="-.", linewidth=1.2, label=f"Short Entry (+{entry_thresh})")
    ax1.axhline(+exit_thresh, color="#4393c3", linestyle=":", linewidth=1.0, label=f"Exit Upper (+{exit_thresh})")
    ax1.axhline(0.0, color="#999999", linestyle="-", linewidth=0.8, alpha=0.7)
    ax1.axhline(-exit_thresh, color="#4393c3", linestyle=":", linewidth=1.0, label=f"Exit Lower (-{exit_thresh})")
    ax1.axhline(-entry_thresh, color="#2166ac", linestyle="-.", linewidth=1.2, label=f"Long Entry (-{entry_thresh})")
    ax1.axhline(-stop_loss_thresh, color="#b2182b", linestyle="--", linewidth=1.2, label=f"Stop-Loss (-{stop_loss_thresh})")

    # Exit corridor shaded region
    ax1.axhspan(-exit_thresh, exit_thresh, color="#d1e5f0", alpha=0.35, label="Mean-Reversion Exit Region")

    # Overlay Trade Entry and Exit markers
    if not trades_df.empty:
        # Long entries
        long_entries = trades_df[trades_df["Direction"] == "LONG"]
        if not long_entries.empty:
            ax1.scatter(
                pd.to_datetime(long_entries["Entry_Date"]),
                long_entries["Entry_Z"],
                color="#008837",
                marker="^",
                s=90,
                zorder=5,
                label=f"Long Entry (n={len(long_entries)})",
            )

        # Short entries
        short_entries = trades_df[trades_df["Direction"] == "SHORT"]
        if not short_entries.empty:
            ax1.scatter(
                pd.to_datetime(short_entries["Entry_Date"]),
                short_entries["Entry_Z"],
                color="#e66101",
                marker="v",
                s=90,
                zorder=5,
                label=f"Short Entry (n={len(short_entries)})",
            )

        # Take-profit exits
        tp_exits = trades_df[trades_df["Exit_Reason"] == "TAKE_PROFIT"]
        if not tp_exits.empty:
            ax1.scatter(
                pd.to_datetime(tp_exits["Exit_Date"]),
                tp_exits["Exit_Z"],
                color="#5e3c99",
                marker="o",
                s=70,
                zorder=5,
                label=f"Take-Profit Exit (n={len(tp_exits)})",
            )

        # Stop-loss exits
        sl_exits = trades_df[trades_df["Exit_Reason"] == "STOP_LOSS"]
        if not sl_exits.empty:
            ax1.scatter(
                pd.to_datetime(sl_exits["Exit_Date"]),
                sl_exits["Exit_Z"],
                color="#b2182b",
                marker="X",
                s=80,
                zorder=5,
                label=f"Stop-Loss Exit (n={len(sl_exits)})",
            )

    ax1.set_title(f"Phase 5: Standardised Spread and Trade Execution Overlay — {pair_name}", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Standardised Spread (Z-Score)", fontsize=10)
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)

    # --- Bottom Panel: Position State ---
    pos = daily_df["Position"]
    ax2.step(dates, pos, where="mid", color="#333333", linewidth=1.2)
    ax2.fill_between(dates, pos, 0, where=(pos > 0), color="#2ca02c", alpha=0.3, step="mid", label="Long Spread (+1)")
    ax2.fill_between(dates, pos, 0, where=(pos < 0), color="#d62728", alpha=0.3, step="mid", label="Short Spread (-1)")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Short (-1)", "Flat (0)", "Long (+1)"])
    ax2.set_ylabel("Position", fontsize=10)
    ax2.set_xlabel("Trading Date", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_pairs_spread_summary(
    pair_daily_dict: Dict[str, pd.DataFrame],
    pair_trades_dict: Dict[str, pd.DataFrame],
    output_path: str,
):
    """Plot multi-panel spread chart for all 3 pairs."""
    set_plot_style()
    n_pairs = len(pair_daily_dict)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 3.5 * n_pairs), dpi=300, sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, (pair, df) in zip(axes, pair_daily_dict.items()):
        dates = pd.to_datetime(df["Date"])
        z = df["Z_Score"]
        trades_df = pair_trades_dict[pair]

        ax.plot(dates, z, color="#2b5c8f", linewidth=1.0, label="Spread Z-Score")
        ax.axhline(2.0, color="#d6604d", linestyle="--", linewidth=0.9)
        ax.axhline(-2.0, color="#2166ac", linestyle="--", linewidth=0.9)
        ax.axhline(0.5, color="#4393c3", linestyle=":", linewidth=0.8)
        ax.axhline(-0.5, color="#4393c3", linestyle=":", linewidth=0.8)
        ax.axhline(3.5, color="#b2182b", linestyle="-.", linewidth=0.9)
        ax.axhline(-3.5, color="#b2182b", linestyle="-.", linewidth=0.9)
        ax.axhspan(-0.5, 0.5, color="#d1e5f0", alpha=0.3)

        # Plot trades
        if not trades_df.empty:
            tp = trades_df[trades_df["Exit_Reason"] == "TAKE_PROFIT"]
            sl = trades_df[trades_df["Exit_Reason"] == "STOP_LOSS"]
            if not tp.empty:
                ax.scatter(pd.to_datetime(tp["Exit_Date"]), tp["Exit_Z"], color="#5e3c99", marker="o", s=40, label="Take-Profit")
            if not sl.empty:
                ax.scatter(pd.to_datetime(sl["Exit_Date"]), sl["Exit_Z"], color="#b2182b", marker="X", s=50, label="Stop-Loss")

        n_t = len(trades_df)
        net_ret = (df["Net_Equity"].iloc[-1] - 1.0) * 100.0
        ax.set_title(f"{pair} — Trades: {n_t}, Net Return: {net_ret:+.2f}%", fontsize=11, fontweight="bold")
        ax.set_ylabel("Z-Score", fontsize=9)
        ax.grid(True, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    axes[-1].set_xlabel("Trading Date", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_performance_summary_bar(
    metrics_df: pd.DataFrame,
    output_path: str,
):
    """Bar chart comparing Net Cumulative Return, Sharpe Ratio, Max Drawdown, and Win Rate."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    pairs = metrics_df["Pair"].tolist()
    clean_labels = [p.replace(".NS", "").replace("COMBINED_PORTFOLIO (Equal Weight)", "Portfolio") for p in pairs]
    x = np.arange(len(pairs))

    # 1. Cumulative Return (%)
    ax = axes[0, 0]
    vals = metrics_df["Cumulative_Net_Return"] * 100.0
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    bars = ax.bar(x, vals, color=bar_colors, width=0.55, edgecolor="#333333", linewidth=0.7)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Cumulative Net Return (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Return (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=15, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:+.1f}%", ha="center", va=va, fontsize=8, fontweight="bold")

    # 2. Sharpe Ratio (rf = 6.5%)
    ax = axes[0, 1]
    vals = metrics_df["Sharpe_Ratio_rf_6_5pct"]
    bar_colors = ["#1f77b4" if v >= 0 else "#b2182b" for v in vals]
    bars = ax.bar(x, vals, color=bar_colors, width=0.55, edgecolor="#333333", linewidth=0.7)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Annualised Sharpe Ratio (rf = 6.5%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=15, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}", ha="center", va=va, fontsize=8, fontweight="bold")

    # 3. Maximum Drawdown Net (%)
    ax = axes[1, 0]
    vals = metrics_df["Max_Drawdown_Net"] * 100.0
    bars = ax.bar(x, vals, color="#e66101", width=0.55, edgecolor="#333333", linewidth=0.7)
    ax.set_title("Maximum Net Drawdown (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Max Drawdown (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=15, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 4. Win Rate (%)
    ax = axes[1, 1]
    vals = metrics_df["Win_Rate"] * 100.0
    bars = ax.bar(x, vals, color="#5e3c99", width=0.55, edgecolor="#333333", linewidth=0.7)
    ax.axhline(50.0, color="#777777", linestyle=":", linewidth=0.8, label="50% Breakeven")
    ax.set_title("Trade Win Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Win Rate (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
