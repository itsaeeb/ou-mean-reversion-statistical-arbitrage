"""
Preprocessing Module for OU Pairs Trading Project.

Handles:
- Data validation (duplicate dates, negative/zero prices, missing values)
- Temporal alignment across all assets
- Log-price transformation: ln(P_t)
- Train-test splitting (Train: 2019-01-01 to 2023-12-31, Test: 2024-01-01 to 2026-09-30)
- Generation of data quality metrics and publication-grade visualizations
"""

import os
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_and_align_prices(
    df_prices: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Validate, clean, and align daily price data across common trading dates.

    Checks performed:
    1. Duplicate index dates.
    2. Missing / NaN values per asset.
    3. Non-positive price values (<= 0).
    4. Common trading date intersection without silent imputation.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame of raw prices with DatetimeIndex and ticker columns.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, any]]
        - Cleaned and aligned DataFrame.
        - Dictionary summarizing data audit results.
    """
    audit: Dict[str, any] = {}
    audit["raw_rows"] = len(df_prices)
    audit["raw_start"] = str(df_prices.index.min().date()) if len(df_prices) > 0 else None
    audit["raw_end"] = str(df_prices.index.max().date()) if len(df_prices) > 0 else None

    # 1. Check for duplicate dates
    duplicate_dates = df_prices.index[df_prices.index.duplicated()].tolist()
    audit["duplicate_date_count"] = len(duplicate_dates)
    audit["duplicate_dates"] = [str(d.date()) for d in duplicate_dates]

    cleaned_df = df_prices[~df_prices.index.duplicated(keep="first")].sort_index()

    # 2. Check for missing values before dropping
    missing_counts = cleaned_df.isnull().sum().to_dict()
    audit["missing_values_per_ticker"] = missing_counts
    audit["total_missing_values"] = int(cleaned_df.isnull().sum().sum())

    # 3. Check for non-positive or invalid prices
    invalid_mask = (cleaned_df <= 0)
    invalid_counts = invalid_mask.sum().to_dict()
    audit["non_positive_price_counts"] = invalid_counts
    if invalid_mask.any().any():
        raise ValueError("Invalid prices detected (values <= 0). Please inspect raw data.")

    # 4. Align on common trading dates (drop dates where any ticker is missing)
    # Per requirements: Do not silently forward-fill missing observations.
    aligned_df = cleaned_df.dropna()
    dropped_rows = len(cleaned_df) - len(aligned_df)
    audit["dropped_unaligned_rows"] = dropped_rows
    audit["aligned_rows"] = len(aligned_df)
    audit["aligned_start"] = str(aligned_df.index.min().date())
    audit["aligned_end"] = str(aligned_df.index.max().date())

    return aligned_df, audit


def compute_log_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the natural logarithm of asset prices: ln(P_t).

    Parameters
    ----------
    df_prices : pd.DataFrame
        Cleaned positive asset prices.

    Returns
    -------
    pd.DataFrame
        Log-price series ln(P_t).
    """
    if (df_prices <= 0).any().any():
        raise ValueError("Cannot calculate logarithm for non-positive prices.")

    log_df = np.log(df_prices)
    if log_df.isin([np.nan, np.inf, -np.inf]).any().any():
        raise ValueError("Log calculation produced invalid numerical values (NaN or Inf).")

    return log_df


def split_train_test(
    df: pd.DataFrame,
    train_start: str = "2019-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2026-09-30",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset into in-sample training and out-of-sample testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned price or log-price DataFrame with DatetimeIndex.
    train_start : str
        Training start date (inclusive).
    train_end : str
        Training end date (inclusive).
    test_start : str
        Testing start date (inclusive).
    test_end : str
        Testing end date (inclusive).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - train_df: in-sample dataset.
        - test_df: out-of-sample dataset.
    """
    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:test_end].copy()

    if train_df.empty:
        raise ValueError(f"Training split is empty for range {train_start} to {train_end}.")
    if test_df.empty:
        raise ValueError(f"Testing split is empty for range {test_start} to {test_end}.")

    return train_df, test_df


def build_data_quality_summary(
    df_clean: pd.DataFrame,
    df_log: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ticker_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Compile comprehensive statistical and data quality metrics for each asset.

    Returns
    -------
    pd.DataFrame
        Statistical summary table per asset.
    """
    records = []
    for col in df_clean.columns:
        name = ticker_names.get(col, col) if ticker_names else col
        s_price = df_clean[col]
        s_log = df_log[col]
        s_train = df_train[col]
        s_test = df_test[col]

        # Daily percentage returns for volatility calculation (252 trading days/year)
        daily_ret = s_price.pct_change().dropna()
        ann_return = daily_ret.mean() * 252 * 100
        ann_vol = daily_ret.std() * np.sqrt(252) * 100

        records.append({
            "Ticker": col,
            "Asset Name": name,
            "Total Observations": len(s_price),
            "Train Obs (2019-2023)": len(s_train),
            "Test Obs (2024-2026)": len(s_test),
            "Min Price (INR)": round(s_price.min(), 2),
            "Max Price (INR)": round(s_price.max(), 2),
            "Mean Price (INR)": round(s_price.mean(), 2),
            "Std Price (INR)": round(s_price.std(), 2),
            "Min Log-Price": round(s_log.min(), 4),
            "Max Log-Price": round(s_log.max(), 4),
            "Mean Log-Price": round(s_log.mean(), 4),
            "Annualised Return (%)": round(ann_return, 2),
            "Annualised Volatility (%)": round(ann_vol, 2),
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def plot_price_series(
    df_prices: pd.DataFrame,
    df_log_prices: pd.DataFrame,
    split_date: str = "2024-01-01",
    save_dir: str = "results/figures",
    ticker_names: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    """
    Create publication-quality plots:
    1. Raw/adjusted daily asset prices.
    2. Normalized asset prices (rebased to 100 at base date).
    3. Natural logarithm of asset prices: ln(P_t).

    Each figure features a distinct vertical boundary marking the train/test split.
    """
    os.makedirs(save_dir, exist_ok=True)
    split_dt = pd.to_datetime(split_date)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # -------------------------------------------------------------
    # Figure 1: Asset Prices (INR)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for idx, col in enumerate(df_prices.columns):
        label = ticker_names.get(col, col) if ticker_names else col
        ax.plot(df_prices.index, df_prices[col], label=label, color=color_palette[idx % len(color_palette)], linewidth=1.5)

    ax.axvline(split_dt, color="#333333", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Train/Test Split ({split_date})")
    ax.text(split_dt - pd.Timedelta(days=20), ax.get_ylim()[1] * 0.95, "In-Sample Training\n(2019 - 2023)",
            horizontalalignment="right", verticalalignment="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8, edgecolor="#999999"))
    ax.text(split_dt + pd.Timedelta(days=20), ax.get_ylim()[1] * 0.95, "Out-of-Sample Testing\n(2024 - 2026)",
            horizontalalignment="left", verticalalignment="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", alpha=0.8, edgecolor="#2b8cbe"))

    ax.set_title("Nifty Bank Constituents: Daily Adjusted Closing Prices (2019 - 2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Adjusted Close Price (INR)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig1_path = os.path.join(save_dir, "asset_prices.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Normalized Asset Prices (Base = 100)
    # -------------------------------------------------------------
    norm_prices = (df_prices / df_prices.iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for idx, col in enumerate(norm_prices.columns):
        label = ticker_names.get(col, col) if ticker_names else col
        ax.plot(norm_prices.index, norm_prices[col], label=label, color=color_palette[idx % len(color_palette)], linewidth=1.5)

    ax.axvline(split_dt, color="#333333", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Train/Test Split ({split_date})")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Nifty Bank Constituents: Normalized Historical Price Trajectories (Base = 100 at 2019-01-01)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Price Index (Base = 100)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig2_path = os.path.join(save_dir, "normalized_asset_prices.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Log-Prices ln(P_t)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for idx, col in enumerate(df_log_prices.columns):
        label = ticker_names.get(col, col) if ticker_names else col
        ax.plot(df_log_prices.index, df_log_prices[col], label=label, color=color_palette[idx % len(color_palette)], linewidth=1.5)

    ax.axvline(split_dt, color="#333333", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Train/Test Split ({split_date})")
    ax.text(split_dt - pd.Timedelta(days=20), ax.get_ylim()[1] * 0.98, "Training Window\n(2019-01-01 to 2023-12-31)",
            horizontalalignment="right", verticalalignment="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8, edgecolor="#999999"))
    ax.text(split_dt + pd.Timedelta(days=20), ax.get_ylim()[1] * 0.98, "Testing Window\n(2024-01-01 to 2026-09-30)",
            horizontalalignment="left", verticalalignment="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", alpha=0.8, edgecolor="#2b8cbe"))

    ax.set_title("Nifty Bank Constituents: Log-Prices ln(P_t) Comparison (2019 - 2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Natural Log-Price ln(P_t)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig3_path = os.path.join(save_dir, "log_prices.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    return fig1_path, fig2_path, fig3_path
