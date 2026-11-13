"""Performance Analytics Module for Quantitative Strategy Evaluation.

Calculates comprehensive risk-adjusted performance metrics, drawdown analysis,
and trade statistics adhering strictly to institutional standards and the
course requirements of MAC F244.

Conventions:
- Annualization factor: 252 trading days.
- Compound annualization: (1 + R_cum) ** (252 / N) - 1.
- Risk-free rates: r_f = 0.0 (unhedged) and r_f = 0.065 (6.5% Indian sovereign benchmark).
- Semi-deviation / downside volatility computed relative to r_f / 252.
- Maximum drawdown measured peak-to-trough on cumulative equity curve.
- Portfolio turnover computed on gross traded dollar volume per annum.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def compute_drawdown_series(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Compute cumulative equity, underwater drawdown series, max drawdown, and max duration.

    Args:
        returns: Array of daily percentage returns (e.g., 0.01 for 1%).

    Returns:
        equity: Cumulative wealth trajectory starting at 1.0.
        drawdown: Underwater drawdown series (positive fraction, e.g. 0.05 for 5% DD).
        max_drawdown: Maximum peak-to-trough decline.
        max_duration: Longest duration in trading days spent underwater.
    """
    if len(returns) == 0:
        return np.array([1.0]), np.array([0.0]), 0.0, 0

    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak

    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Max drawdown duration
    duration = 0
    max_duration = 0
    for dd in drawdown:
        if dd > 1e-6:
            duration += 1
            if duration > max_duration:
                max_duration = duration
        else:
            duration = 0

    return equity, drawdown, max_dd, max_duration


def compute_advanced_metrics(
    net_returns: np.ndarray,
    gross_returns: np.ndarray,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.065,
    ann_factor: int = 252,
) -> Dict[str, Any]:
    """Calculate institutional strategy performance and risk metrics.

    Args:
        net_returns: Array of daily net returns.
        gross_returns: Array of daily gross returns.
        trades_df: DataFrame of completed trades.
        risk_free_rate: Annualized risk-free benchmark rate.
        ann_factor: Number of trading days per year.

    Returns:
        Dictionary of standardized performance metrics.
    """
    n = len(net_returns)
    years = n / ann_factor if n > 0 else 1.0
    daily_rf = risk_free_rate / ann_factor

    # 1. Cumulative & Annualized Returns
    cum_net = float(np.prod(1.0 + net_returns) - 1.0) if n > 0 else 0.0
    cum_gross = float(np.prod(1.0 + gross_returns) - 1.0) if n > 0 else 0.0

    if cum_net > -1.0 and years > 0:
        ann_net = float((1.0 + cum_net) ** (1.0 / years) - 1.0)
    else:
        ann_net = -1.0

    if cum_gross > -1.0 and years > 0:
        ann_gross = float((1.0 + cum_gross) ** (1.0 / years) - 1.0)
    else:
        ann_gross = -1.0

    # 2. Volatilities & Downside Deviation
    daily_std = float(np.std(net_returns, ddof=1)) if n > 1 else 0.0
    ann_vol = float(daily_std * np.sqrt(ann_factor))

    # Downside semi-deviation (returns below daily risk-free rate)
    downside_diffs = np.minimum(0.0, net_returns - daily_rf)
    downside_var = float(np.mean(downside_diffs ** 2)) if n > 0 else 0.0
    downside_vol = float(np.sqrt(downside_var) * np.sqrt(ann_factor))

    # Zero-rf downside semi-deviation
    downside_zero = np.minimum(0.0, net_returns)
    downside_zero_var = float(np.mean(downside_zero ** 2)) if n > 0 else 0.0
    downside_zero_vol = float(np.sqrt(downside_zero_var) * np.sqrt(ann_factor))

    # 3. Sharpe & Sortino Ratios
    sharpe_zero = float(ann_net / ann_vol) if ann_vol > 1e-6 else 0.0
    sharpe_rf = float((ann_net - risk_free_rate) / ann_vol) if ann_vol > 1e-6 else 0.0

    sortino_zero = float(ann_net / downside_zero_vol) if downside_zero_vol > 1e-6 else 0.0
    sortino_rf = float((ann_net - risk_free_rate) / downside_vol) if downside_vol > 1e-6 else 0.0

    # 4. Drawdowns
    _, _, max_dd_net, max_dd_days_net = compute_drawdown_series(net_returns)
    _, _, max_dd_gross, max_dd_days_gross = compute_drawdown_series(gross_returns)

    # Calmar Ratio: Annualized Net Return / Max Drawdown
    calmar_ratio = float(ann_net / max_dd_net) if max_dd_net > 1e-4 else 0.0

    # 5. Value-at-Risk (Historical 95% and 99%)
    if n > 20:
        var_95 = float(np.percentile(net_returns, 5.0))
        var_99 = float(np.percentile(net_returns, 1.0))
        cvar_95 = float(np.mean(net_returns[net_returns <= var_95]))
    else:
        var_95 = 0.0
        var_99 = 0.0
        cvar_95 = 0.0

    # 6. Trade Activity & Diagnostics
    num_trades = len(trades_df)
    if num_trades > 0:
        wins = int((trades_df["Net_PnL"] > 0).sum())
        losses = int((trades_df["Net_PnL"] <= 0).sum())
        win_rate = float(wins / num_trades)
        avg_holding = float(trades_df["Holding_Days"].mean())
        total_costs = float(trades_df["Transaction_Costs"].sum())

        long_trades = int((trades_df["Direction"] == "LONG").sum()) if "Direction" in trades_df else 0
        short_trades = int((trades_df["Direction"] == "SHORT").sum()) if "Direction" in trades_df else 0

        # Stop-loss vs take-profit counts
        if "Exit_Reason" in trades_df:
            stop_exits = int((trades_df["Exit_Reason"] == "Stop-Loss").sum())
            take_profit_exits = int((trades_df["Exit_Reason"] == "Take-Profit").sum())
        else:
            stop_exits = 0
            take_profit_exits = 0

        # PnL distribution
        win_pnls = trades_df[trades_df["Net_PnL"] > 0]["Net_PnL"]
        loss_pnls = trades_df[trades_df["Net_PnL"] <= 0]["Net_PnL"]

        avg_win = float(win_pnls.mean()) if len(win_pnls) > 0 else 0.0
        avg_loss = float(loss_pnls.mean()) if len(loss_pnls) > 0 else 0.0

        sum_gains = float(win_pnls.sum()) if len(win_pnls) > 0 else 0.0
        sum_losses = float(abs(loss_pnls.sum())) if len(loss_pnls) > 0 else 0.0
        profit_factor = float(sum_gains / sum_losses) if sum_losses > 1e-6 else (99.0 if sum_gains > 0 else 0.0)
        payoff_ratio = float(avg_win / abs(avg_loss)) if abs(avg_loss) > 1e-6 else 0.0

        # Annualized turnover: 2 turns (buy + sell) per trade
        turnover = float((num_trades * 2.0) / years)
    else:
        wins = 0
        losses = 0
        win_rate = 0.0
        avg_holding = 0.0
        total_costs = 0.0
        long_trades = 0
        short_trades = 0
        stop_exits = 0
        take_profit_exits = 0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        payoff_ratio = 0.0
        turnover = 0.0

    cost_drag = float(ann_gross - ann_net)

    return {
        "Total_Days": n,
        "Years": round(years, 3),
        "Cumulative_Gross_Return": round(cum_gross, 4),
        "Cumulative_Net_Return": round(cum_net, 4),
        "Annualised_Gross_Return": round(ann_gross, 4),
        "Annualised_Net_Return": round(ann_net, 4),
        "Annualised_Volatility": round(ann_vol, 4),
        "Downside_Volatility_rf": round(downside_vol, 4),
        "Sharpe_Ratio_rf_zero": round(sharpe_zero, 4),
        "Sharpe_Ratio_rf_6_5pct": round(sharpe_rf, 4),
        "Sortino_Ratio_rf_zero": round(sortino_zero, 4),
        "Sortino_Ratio_rf_6_5pct": round(sortino_rf, 4),
        "Calmar_Ratio": round(calmar_ratio, 4),
        "Max_Drawdown_Gross": round(max_dd_gross, 4),
        "Max_Drawdown_Net": round(max_dd_net, 4),
        "Max_Drawdown_Duration_Days": max_dd_days_net,
        "Daily_VaR_95pct": round(var_95, 4),
        "Daily_CVaR_95pct": round(cvar_95, 4),
        "Total_Trades": num_trades,
        "Winning_Trades": wins,
        "Losing_Trades": losses,
        "Win_Rate": round(win_rate, 4),
        "Long_Trades": long_trades,
        "Short_Trades": short_trades,
        "Take_Profit_Exits": take_profit_exits,
        "Stop_Loss_Exits": stop_exits,
        "Avg_Holding_Days": round(avg_holding, 1),
        "Avg_Win_Return": round(avg_win, 4),
        "Avg_Loss_Return": round(avg_loss, 4),
        "Payoff_Ratio": round(payoff_ratio, 4),
        "Profit_Factor": round(profit_factor, 4),
        "Total_Transaction_Costs": round(total_costs, 4),
        "Cost_Drag_Annualized": round(cost_drag, 4),
        "Annualised_Turnover": round(turnover, 2),
    }

