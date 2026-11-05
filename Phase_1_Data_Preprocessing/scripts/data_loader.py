"""
Data Loader Module for OU Pairs Trading Project.

Handles downloading daily historical price data for Nifty Bank constituents
using yfinance and saving raw datasets for reproducible research.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf

# Constituents of interest from the Nifty Bank index
DEFAULT_TICKERS: Dict[str, str] = {
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "SBIN.NS": "State Bank of India",
    "INDUSINDBK.NS": "IndusInd Bank",
}

DEFAULT_START_DATE: str = "2019-01-01"
# End date is exclusive in yfinance, so '2026-10-01' captures through 2026-09-30
DEFAULT_END_DATE: str = "2026-10-01"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_1_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_RAW_DIR = os.path.join(PHASE_1_DIR, "data", "raw")


def download_historical_data(
    tickers: Optional[List[str]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    save_raw: bool = True,
    raw_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily OHLCV and Adjusted Close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : Optional[List[str]]
        List of Yahoo Finance ticker symbols. Defaults to DEFAULT_TICKERS keys.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format (exclusive).
    save_raw : bool
        Whether to save raw data to CSV.
    raw_dir : str
        Directory to store raw CSV data files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - full_raw_df: MultiIndex DataFrame containing all OHLCV columns.
        - adj_close_df: Single-index DataFrame of daily Adjusted Close prices.
    """
    if tickers is None:
        tickers = list(DEFAULT_TICKERS.keys())

    print(f"Downloading historical data for {len(tickers)} tickers from {start_date} to {end_date}...")
    # auto_adjust=False preserves the standard Close and explicit Adj Close columns
    raw_data: pd.DataFrame = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if raw_data.empty:
        raise ValueError("Download failed or returned an empty dataset. Check tickers and date range.")

    # Ensure index is datetime and normalize timezone to tz-naive
    raw_data.index = pd.to_datetime(raw_data.index)
    if raw_data.index.tz is not None:
        raw_data.index = raw_data.index.tz_localize(None)
    raw_data.index = raw_data.index.normalize()

    # Extract Adjusted Close series
    if "Adj Close" in raw_data.columns:
        adj_close = raw_data["Adj Close"].copy()
    else:
        raise KeyError("Expected 'Adj Close' column not found in downloaded data.")

    # Ensure columns match requested tickers order
    present_tickers = [t for t in tickers if t in adj_close.columns]
    missing_tickers = [t for t in tickers if t not in adj_close.columns]
    if missing_tickers:
        print(f"Warning: The following tickers were not returned: {missing_tickers}")

    adj_close = adj_close[present_tickers]

    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR

    if save_raw:
        os.makedirs(raw_dir, exist_ok=True)
        raw_full_path = os.path.join(raw_dir, "raw_market_data.csv")
        raw_adj_path = os.path.join(raw_dir, "raw_adj_close.csv")

        # Save multi-index raw data
        raw_data.to_csv(raw_full_path)
        # Save raw adjusted close prices
        adj_close.to_csv(raw_adj_path)
        print(f"Saved full raw market data to: {raw_full_path}")
        print(f"Saved raw adjusted close prices to: {raw_adj_path}")

    return raw_data, adj_close


def load_raw_data(
    filepath: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load previously saved raw adjusted close data from CSV.

    Parameters
    ----------
    filepath : Optional[str]
        Path to the CSV file. If None, defaults to DEFAULT_RAW_DIR/raw_adj_close.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and ticker columns.
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "raw_adj_close.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df
