"""Sensitivity and Robustness Analysis for OU Pairs Trading Strategy.

Executes 4 comprehensive sensitivity experiments:
1. Threshold Sensitivity Grid:
   - Evaluates a 2D matrix of entry thresholds (1.25 to 3.0) and exit thresholds (0.0 to 1.0).
   - Outputs matrices of Net Return, Sharpe Ratio, Max Drawdown, Win Rate, and Trade Count.
2. Transaction Cost Friction Frontier:
   - Sweeps per-leg transaction costs from 0 to 30 bps (0 to 120 bps round-trip).
   - Identifies the exact break-even transaction cost threshold.
3. Stop-Loss Barrier Sensitivity:
   - Sweeps stop-loss thresholds from 2.5 to 5.0 (and unconstrained inf).
   - Evaluates capital protection versus premature liquidation.
4. Rolling Estimation Window Sensitivity:
   - Compares responsiveness versus noise across calibration windows (60, 90, 120, 180, 250 days).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from Phase_5_Trading_Strategy.scripts.backtest_engine import (
    simulate_pair_strategy,
    simulate_combined_portfolio,
    StrategyConfig,
)
from Phase_6_Performance_Benchmark.scripts.performance_analytics import compute_advanced_metrics


def run_threshold_grid_experiment(
    test_df: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    beta: float,
    pair_name: str,
    z_col: str,
    valid_col: str,
    entry_grid: Optional[List[float]] = None,
    exit_grid: Optional[List[float]] = None,
    stop_loss: float = 3.5,
    cost_per_leg_bps: float = 10.0,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """Run 2D parameter grid sweep over entry and exit thresholds."""
    if entry_grid is None:
        entry_grid = [1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
    if exit_grid is None:
        exit_grid = [0.00, 0.25, 0.50, 0.75, 1.00]

    records = []

    for se in entry_grid:
        for sx in exit_grid:
            if se <= sx + 0.20:
                continue

            cfg = StrategyConfig(
                entry_threshold=se,
                exit_threshold=sx,
                stop_loss_threshold=stop_loss,
                cost_per_leg_bps=cost_per_leg_bps,
                risk_free_rate=risk_free_rate,
            )

            daily_df, trades_df, _ = simulate_pair_strategy(
                df=test_df,
                stock_a=stock_a,
                stock_b=stock_b,
                beta=beta,
                pair_name=pair_name,
                z_col=z_col,
                valid_col=valid_col,
                config=cfg,
            )

            m = compute_advanced_metrics(
                net_returns=daily_df["Net_Return"].values,
                gross_returns=daily_df["Gross_Return"].values,
                trades_df=trades_df,
                risk_free_rate=risk_free_rate,
            )

            records.append({
                "Pair": pair_name,
                "Entry_Threshold": se,
                "Exit_Threshold": sx,
                "Stop_Loss": stop_loss,
                "Cost_Bps_Per_Leg": cost_per_leg_bps,
                "Total_Trades": m["Total_Trades"],
                "Win_Rate": m["Win_Rate"],
                "Avg_Holding_Days": m["Avg_Holding_Days"],
                "Cumulative_Net_Return": m["Cumulative_Net_Return"],
                "Annualised_Net_Return": m["Annualised_Net_Return"],
                "Annualised_Volatility": m["Annualised_Volatility"],
                "Sharpe_Ratio_rf_zero": m["Sharpe_Ratio_rf_zero"],
                "Sharpe_Ratio_rf_6_5pct": m["Sharpe_Ratio_rf_6_5pct"],
                "Max_Drawdown_Net": m["Max_Drawdown_Net"],
                "Profit_Factor": m["Profit_Factor"],
            })

    return pd.DataFrame(records)


def run_cost_sensitivity_experiment(
    test_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    cost_grid_bps: Optional[List[float]] = None,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 3.5,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """Evaluate strategy performance across varying transaction cost regimes."""
    if cost_grid_bps is None:
        cost_grid_bps = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]

    records = []

    for cost_bps in cost_grid_bps:
        cfg = StrategyConfig(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss,
            cost_per_leg_bps=cost_bps,
            risk_free_rate=risk_free_rate,
        )

        pair_dailies = {}
        pair_trades = {}

        for _, row in pairs_df.iterrows():
            pair = row["Pair"]
            sa = row["Stock A (Y)"]
            sb = row["Stock B (X)"]
            beta = float(row["Hedge Ratio (Beta)"])

            daily_df, trades_df, _ = simulate_pair_strategy(
                df=test_df,
                stock_a=sa,
                stock_b=sb,
                beta=beta,
                pair_name=pair,
                z_col=f"Z_Score_{pair}",
                valid_col=f"Is_Valid_{pair}",
                config=cfg,
            )

            m = compute_advanced_metrics(
                net_returns=daily_df["Net_Return"].values,
                gross_returns=daily_df["Gross_Return"].values,
                trades_df=trades_df,
                risk_free_rate=risk_free_rate,
            )

            records.append({
                "Pair": pair,
                "Cost_Bps_Per_Leg": cost_bps,
                "Round_Trip_Cost_Bps": 4.0 * cost_bps,
                "Cumulative_Gross_Return": m["Cumulative_Gross_Return"],
                "Cumulative_Net_Return": m["Cumulative_Net_Return"],
                "Annualised_Net_Return": m["Annualised_Net_Return"],
                "Sharpe_Ratio_rf_zero": m["Sharpe_Ratio_rf_zero"],
                "Sharpe_Ratio_rf_6_5pct": m["Sharpe_Ratio_rf_6_5pct"],
                "Max_Drawdown_Net": m["Max_Drawdown_Net"],
                "Total_Transaction_Costs": m["Total_Transaction_Costs"],
                "Total_Trades": m["Total_Trades"],
            })

            pair_dailies[pair] = daily_df
            pair_trades[pair] = trades_df

        # Portfolio combined under this cost regime
        port_daily, port_trades, _ = simulate_combined_portfolio(
            pair_daily_dict=pair_dailies,
            pair_trades_dict=pair_trades,
            config=cfg,
        )

        port_m = compute_advanced_metrics(
            net_returns=port_daily["Portfolio_Net_Return"].values,
            gross_returns=port_daily["Portfolio_Gross_Return"].values,
            trades_df=port_trades,
            risk_free_rate=risk_free_rate,
        )

        records.append({
            "Pair": "COMBINED PORTFOLIO",
            "Cost_Bps_Per_Leg": cost_bps,
            "Round_Trip_Cost_Bps": 4.0 * cost_bps,
            "Cumulative_Gross_Return": port_m["Cumulative_Gross_Return"],
            "Cumulative_Net_Return": port_m["Cumulative_Net_Return"],
            "Annualised_Net_Return": port_m["Annualised_Net_Return"],
            "Sharpe_Ratio_rf_zero": port_m["Sharpe_Ratio_rf_zero"],
            "Sharpe_Ratio_rf_6_5pct": port_m["Sharpe_Ratio_rf_6_5pct"],
            "Max_Drawdown_Net": port_m["Max_Drawdown_Net"],
            "Total_Transaction_Costs": port_m["Total_Transaction_Costs"],
            "Total_Trades": port_m["Total_Trades"],
        })

    return pd.DataFrame(records)


def run_stop_loss_sensitivity_experiment(
    test_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    stop_grid: Optional[List[float]] = None,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    cost_per_leg_bps: float = 10.0,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """Evaluate strategy resilience across various stop-loss barrier levels."""
    if stop_grid is None:
        stop_grid = [2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, 99.0]

    records = []

    for stop_z in stop_grid:
        cfg = StrategyConfig(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_z,
            cost_per_leg_bps=cost_per_leg_bps,
            risk_free_rate=risk_free_rate,
        )

        pair_dailies = {}
        pair_trades = {}

        for _, row in pairs_df.iterrows():
            pair = row["Pair"]
            sa = row["Stock A (Y)"]
            sb = row["Stock B (X)"]
            beta = float(row["Hedge Ratio (Beta)"])

            daily_df, trades_df, _ = simulate_pair_strategy(
                df=test_df,
                stock_a=sa,
                stock_b=sb,
                beta=beta,
                pair_name=pair,
                z_col=f"Z_Score_{pair}",
                valid_col=f"Is_Valid_{pair}",
                config=cfg,
            )

            m = compute_advanced_metrics(
                net_returns=daily_df["Net_Return"].values,
                gross_returns=daily_df["Gross_Return"].values,
                trades_df=trades_df,
                risk_free_rate=risk_free_rate,
            )

            records.append({
                "Pair": pair,
                "Stop_Loss_Z": stop_z if stop_z < 90.0 else "None",
                "Total_Trades": m["Total_Trades"],
                "Stop_Loss_Exits": m["Stop_Loss_Exits"],
                "Take_Profit_Exits": m["Take_Profit_Exits"],
                "Win_Rate": m["Win_Rate"],
                "Cumulative_Net_Return": m["Cumulative_Net_Return"],
                "Annualised_Net_Return": m["Annualised_Net_Return"],
                "Sharpe_Ratio_rf_zero": m["Sharpe_Ratio_rf_zero"],
                "Max_Drawdown_Net": m["Max_Drawdown_Net"],
            })

            pair_dailies[pair] = daily_df
            pair_trades[pair] = trades_df

        # Portfolio
        port_daily, port_trades, _ = simulate_combined_portfolio(
            pair_daily_dict=pair_dailies,
            pair_trades_dict=pair_trades,
            config=cfg,
        )

        port_m = compute_advanced_metrics(
            net_returns=port_daily["Portfolio_Net_Return"].values,
            gross_returns=port_daily["Portfolio_Gross_Return"].values,
            trades_df=port_trades,
            risk_free_rate=risk_free_rate,
        )

        records.append({
            "Pair": "COMBINED PORTFOLIO",
            "Stop_Loss_Z": stop_z if stop_z < 90.0 else "None",
            "Total_Trades": port_m["Total_Trades"],
            "Stop_Loss_Exits": port_m["Stop_Loss_Exits"],
            "Take_Profit_Exits": port_m["Take_Profit_Exits"],
            "Win_Rate": port_m["Win_Rate"],
            "Cumulative_Net_Return": port_m["Cumulative_Net_Return"],
            "Annualised_Net_Return": port_m["Annualised_Net_Return"],
            "Sharpe_Ratio_rf_zero": port_m["Sharpe_Ratio_rf_zero"],
            "Max_Drawdown_Net": port_m["Max_Drawdown_Net"],
        })

    return pd.DataFrame(records)

