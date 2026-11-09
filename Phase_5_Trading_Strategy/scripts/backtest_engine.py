"""Backtest Engine for OU-Based Pairs Trading Strategy.

Implements:
1. Spread gross return calculation with gross exposure normalization:
   r_spread = (r_A - beta * r_B) / (1 + |beta|)
2. Mean-reversion trading strategy state machine with configurable thresholds:
   - Long Spread entry: s_t <= -entry_threshold (default -2.0)
   - Short Spread entry: s_t >= +entry_threshold (default +2.0)
   - Take-Profit exit: |s_t| <= exit_threshold (default 0.5)
   - Stop-Loss exit: |s_t| >= stop_loss_threshold (default 3.5)
3. Strict zero look-ahead execution:
   - Signal generated at close of day t.
   - Execution for session t+1.
   - Returns earned during session t+1.
4. Transaction cost accounting:
   - 10 bps per traded leg per transaction.
   - Entry: 2 legs * 10 bps = 20 bps (0.20%).
   - Exit: 2 legs * 10 bps = 20 bps (0.20%).
   - Total round-trip cost = 40 bps (0.40%).
5. Single-pair and combined portfolio performance metrics and equity tracking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration parameters for the OU pairs trading strategy."""
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = 3.5
    cost_per_leg_bps: float = 10.0  # 10 bps = 0.10% = 0.0010
    risk_free_rate: float = 0.065   # 6.5% annual risk-free rate
    annualization_factor: int = 252

    @property
    def cost_per_leg(self) -> float:
        return self.cost_per_leg_bps / 10000.0

    @property
    def entry_cost(self) -> float:
        """2 legs traded on entry (Stock A and Stock B)."""
        return 2.0 * self.cost_per_leg

    @property
    def exit_cost(self) -> float:
        """2 legs traded on exit (Stock A and Stock B)."""
        return 2.0 * self.cost_per_leg

    @property
    def round_trip_cost(self) -> float:
        """Total round-trip cost for opening and closing a pair position."""
        return self.entry_cost + self.exit_cost


def compute_spread_return(ret_a: float, ret_b: float, beta: float) -> float:
    """Compute normalized gross spread return for long spread position.

    Formula:
        r_spread = (r_A - beta * r_B) / (1 + |beta|)

    Args:
        ret_a: Return of Stock A.
        ret_b: Return of Stock B.
        beta: Hedge ratio from training OLS regression.

    Returns:
        Normalized gross return of long spread position.
    """
    if np.isnan(beta) or beta == 0:
        raise ValueError(f"Invalid hedge ratio beta: {beta}")
    denom = 1.0 + abs(beta)
    return (ret_a - beta * ret_b) / denom


def simulate_pair_strategy(
    df: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    beta: float,
    pair_name: str,
    z_col: str,
    valid_col: str,
    config: Optional[StrategyConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Simulate the OU pairs trading strategy for a single cointegrated pair.

    Execution timeline (Strict Zero Look-Ahead):
    -------------------------------------------
    - At close of day t-1:
      Signal is evaluated using s_{t-1} and valid flag.
      If flat and s_{t-1} <= -entry_threshold -> Target position for day t = +1 (LONG).
      If flat and s_{t-1} >= +entry_threshold -> Target position for day t = -1 (SHORT).
    - During day t session:
      Position Pos_t is held.
      Earns gross return: Pos_t * (r_{A, t} - beta * r_{B, t}) / (1 + |beta|).
      If day t is entry day, deduct entry cost (20 bps).
    - At close of day t:
      Observe s_t. Check exit criteria:
      - Stop-loss: |s_t| >= stop_loss_threshold
      - Take-profit: |s_t| <= exit_threshold
      - Final day: t == last day of test period
      If exit triggered:
      Deduct exit cost (20 bps) on day t.
      Close trade, set Pos_{t+1} = 0 (FLAT).

    Args:
        df: DataFrame containing Date, stock prices, Z_Score, and Is_Valid.
        stock_a: Ticker name for Stock A.
        stock_b: Ticker name for Stock B.
        beta: Fixed hedge ratio.
        pair_name: Name of the pair.
        z_col: Column name for standardized spread s_t.
        valid_col: Column name for boolean Is_Valid.
        config: StrategyConfig instance.

    Returns:
        daily_df: DataFrame of daily positions, returns, costs, and equity.
        trades_df: DataFrame of all completed trades.
        metrics: Dictionary of summary performance metrics.
    """
    if config is None:
        config = StrategyConfig()

    data = df.copy().reset_index(drop=True)
    n_days = len(data)

    # Compute daily percentage returns of individual stocks
    # Using close-to-close returns
    data["ret_A"] = data[stock_a].pct_change().fillna(0.0)
    data["ret_B"] = data[stock_b].pct_change().fillna(0.0)

    # Compute gross spread return for long spread: (ret_A - beta * ret_B) / (1 + |beta|)
    denom = 1.0 + abs(beta)
    data["ret_spread_long"] = (data["ret_A"] - beta * data["ret_B"]) / denom

    # Simulation state variables
    current_pos = 0  # 0: Flat, +1: Long spread, -1: Short spread
    trades: List[Dict[str, Any]] = []
    active_trade: Optional[Dict[str, Any]] = None

    # Daily tracking series
    daily_positions = np.zeros(n_days, dtype=int)
    daily_gross_returns = np.zeros(n_days, dtype=float)
    daily_costs = np.zeros(n_days, dtype=float)
    daily_net_returns = np.zeros(n_days, dtype=float)
    daily_signals = np.zeros(n_days, dtype=int)  # Signal generated at close of day t
    trade_id_series = np.zeros(n_days, dtype=int)

    # Pending order from yesterday's close
    pending_pos = 0
    pending_signal_info: Optional[Dict[str, Any]] = None

    for t in range(n_days):
        dt = data.loc[t, "Date"]
        p_a = data.loc[t, stock_a]
        p_b = data.loc[t, stock_b]
        z_t = data.loc[t, z_col]
        is_valid_t = bool(data.loc[t, valid_col]) if valid_col in data.columns else True
        ret_sp = data.loc[t, "ret_spread_long"]
        exit_reason = None

        # 1. Execute pending entry order from day t-1 close
        if pending_pos != 0 and current_pos == 0:
            current_pos = pending_pos
            # Deduct entry cost on execution day
            daily_costs[t] += config.entry_cost
            active_trade = {
                "Trade_ID": len(trades) + 1,
                "Pair": pair_name,
                "Stock_A": stock_a,
                "Stock_B": stock_b,
                "Hedge_Ratio_Beta": beta,
                "Direction": "LONG" if current_pos == 1 else "SHORT",
                "Signal_Date": pending_signal_info["signal_date"],
                "Entry_Date": dt,
                "Entry_Price_A": pending_signal_info["entry_price_a"],
                "Entry_Price_B": pending_signal_info["entry_price_b"],
                "Entry_Z": pending_signal_info["entry_z"],
                "Entry_Cost": config.entry_cost,
                "Exit_Cost": 0.0,
                "Holding_Days": 0,
                "Daily_Gross_Returns": [],
                "Daily_Net_Returns": [],
            }
            pending_pos = 0
            pending_signal_info = None

        daily_positions[t] = current_pos
        trade_id_series[t] = active_trade["Trade_ID"] if active_trade else 0

        # 2. Earn return during day t session
        if current_pos != 0:
            gross_ret = current_pos * ret_sp
            daily_gross_returns[t] = gross_ret
            active_trade["Holding_Days"] += 1
            active_trade["Daily_Gross_Returns"].append(gross_ret)

            # 3. Check exit conditions at close of day t
            exit_reason = None
            if is_valid_t and not np.isnan(z_t):
                if abs(z_t) >= config.stop_loss_threshold:
                    exit_reason = "STOP_LOSS"
                elif abs(z_t) <= config.exit_threshold:
                    exit_reason = "TAKE_PROFIT"

            # Mandatory closure on last day of test period
            if t == n_days - 1 and exit_reason is None:
                exit_reason = "PERIOD_END"

            if exit_reason is not None:
                # Deduct exit cost on exit day
                daily_costs[t] += config.exit_cost
                active_trade["Exit_Date"] = dt
                active_trade["Exit_Price_A"] = p_a
                active_trade["Exit_Price_B"] = p_b
                active_trade["Exit_Z"] = z_t
                active_trade["Exit_Reason"] = exit_reason
                active_trade["Exit_Cost"] = config.exit_cost

                # Net return for today incorporates today's costs
                today_net = daily_gross_returns[t] - daily_costs[t]
                daily_net_returns[t] = today_net
                active_trade["Daily_Net_Returns"].append(today_net)

                # Compound trade gross and net returns
                # Trade Gross Return: prod(1 + r_gross) - 1
                trade_gross = np.prod([1.0 + r for r in active_trade["Daily_Gross_Returns"]]) - 1.0
                # Trade Net Return: compounded daily net returns
                trade_net = np.prod([1.0 + r for r in active_trade["Daily_Net_Returns"]]) - 1.0
                total_trade_costs = active_trade["Entry_Cost"] + active_trade["Exit_Cost"]

                active_trade["Gross_PnL"] = float(trade_gross)
                active_trade["Transaction_Costs"] = float(total_trade_costs)
                active_trade["Net_PnL"] = float(trade_net)
                active_trade["Is_Win"] = bool(trade_net > 0.0)

                trades.append(active_trade)
                current_pos = 0
                active_trade = None
            else:
                # Still holding through to tomorrow
                today_net = daily_gross_returns[t] - daily_costs[t]
                daily_net_returns[t] = today_net
                active_trade["Daily_Net_Returns"].append(today_net)
        else:
            # Flat during session t
            daily_net_returns[t] = 0.0 - daily_costs[t]

        # 4. If flat at close of day t and did not just exit today, check entry signal for day t+1
        if current_pos == 0 and exit_reason is None and t < n_days - 1:
            if is_valid_t and not np.isnan(z_t):
                # Valid entry must be between entry threshold and stop-loss threshold
                if -config.stop_loss_threshold < z_t <= -config.entry_threshold:
                    daily_signals[t] = 1  # Long signal
                    pending_pos = 1
                    pending_signal_info = {
                        "signal_date": dt,
                        "entry_price_a": p_a,
                        "entry_price_b": p_b,
                        "entry_z": z_t,
                    }
                elif config.entry_threshold <= z_t < config.stop_loss_threshold:
                    daily_signals[t] = -1  # Short signal
                    pending_pos = -1
                    pending_signal_info = {
                        "signal_date": dt,
                        "entry_price_a": p_a,
                        "entry_price_b": p_b,
                        "entry_z": z_t,
                    }

    # Build daily results DataFrame
    daily_df = pd.DataFrame({
        "Date": data["Date"],
        "Pair": pair_name,
        "Stock_A": stock_a,
        "Stock_B": stock_b,
        "Hedge_Ratio_Beta": beta,
        "Price_A": data[stock_a],
        "Price_B": data[stock_b],
        "Z_Score": data[z_col],
        "Is_Valid": data[valid_col] if valid_col in data.columns else True,
        "Signal": daily_signals,
        "Position": daily_positions,
        "Trade_ID": trade_id_series,
        "Gross_Return": daily_gross_returns,
        "Transaction_Cost": daily_costs,
        "Net_Return": daily_net_returns,
    })

    # Compute daily equity curve: starting from 1.0
    daily_df["Gross_Equity"] = (1.0 + daily_df["Gross_Return"]).cumprod()
    daily_df["Net_Equity"] = (1.0 + daily_df["Net_Return"]).cumprod()

    # Drawdown series
    daily_df["Gross_Peak"] = daily_df["Gross_Equity"].cummax()
    daily_df["Gross_Drawdown"] = (daily_df["Gross_Peak"] - daily_df["Gross_Equity"]) / daily_df["Gross_Peak"]
    daily_df["Net_Peak"] = daily_df["Net_Equity"].cummax()
    daily_df["Net_Drawdown"] = (daily_df["Net_Peak"] - daily_df["Net_Equity"]) / daily_df["Net_Peak"]

    # Build trades DataFrame
    if trades:
        # Remove list fields for clean CSV export
        clean_trades = []
        for tr in trades:
            tr_copy = {k: v for k, v in tr.items() if k not in ("Daily_Gross_Returns", "Daily_Net_Returns")}
            clean_trades.append(tr_copy)
        trades_df = pd.DataFrame(clean_trades)
    else:
        trades_df = pd.DataFrame(columns=[
            "Trade_ID", "Pair", "Stock_A", "Stock_B", "Hedge_Ratio_Beta",
            "Direction", "Signal_Date", "Entry_Date", "Entry_Price_A",
            "Entry_Price_B", "Entry_Z", "Exit_Date", "Exit_Price_A",
            "Exit_Price_B", "Exit_Z", "Exit_Reason", "Entry_Cost",
            "Exit_Cost", "Holding_Days", "Gross_PnL", "Transaction_Costs",
            "Net_PnL", "Is_Win",
        ])

    metrics = calculate_performance_metrics(
        net_returns=daily_df["Net_Return"].values,
        gross_returns=daily_df["Gross_Return"].values,
        trades_df=trades_df,
        config=config,
    )
    metrics["Pair"] = pair_name

    return daily_df, trades_df, metrics


def calculate_performance_metrics(
    net_returns: np.ndarray,
    gross_returns: np.ndarray,
    trades_df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> Dict[str, Any]:
    """Calculate institutional strategy performance metrics.

    Metrics:
    1. Cumulative Return (Gross and Net)
    2. Annualised Return
    3. Annualised Volatility
    4. Annualised Sharpe Ratio (both at rf=0 and rf=config.risk_free_rate)
    5. Maximum Drawdown (Gross and Net)
    6. Win Rate (% of trades with Net PnL > 0)
    7. Total Number of Trades
    8. Average Holding Period (trading days)
    9. Portfolio Turnover (annualized)
    10. Total Transaction Costs

    Args:
        net_returns: Array of daily net returns.
        gross_returns: Array of daily gross returns.
        trades_df: DataFrame of completed trades.
        config: StrategyConfig instance.

    Returns:
        Dictionary of performance metrics.
    """
    if config is None:
        config = StrategyConfig()

    n = len(net_returns)
    ann_factor = config.annualization_factor
    years = n / ann_factor if n > 0 else 1.0

    # Cumulative and compound annual growth
    cum_net = float(np.prod(1.0 + net_returns) - 1.0)
    cum_gross = float(np.prod(1.0 + gross_returns) - 1.0)

    # Annualised return: (1 + cum_ret) ** (1 / years) - 1
    if cum_net > -1.0 and years > 0:
        ann_net_ret = float((1.0 + cum_net) ** (1.0 / years) - 1.0)
    else:
        ann_net_ret = -1.0

    if cum_gross > -1.0 and years > 0:
        ann_gross_ret = float((1.0 + cum_gross) ** (1.0 / years) - 1.0)
    else:
        ann_gross_ret = -1.0

    # Volatility
    daily_std = float(np.std(net_returns, ddof=1)) if n > 1 else 0.0
    ann_vol = float(daily_std * np.sqrt(ann_factor))

    # Sharpe ratios
    rf = config.risk_free_rate
    if ann_vol > 1e-8:
        sharpe_rf = float((ann_net_ret - rf) / ann_vol)
        sharpe_zero = float(ann_net_ret / ann_vol)
    else:
        sharpe_rf = 0.0
        sharpe_zero = 0.0

    # Maximum Drawdown
    net_equity = np.cumprod(1.0 + net_returns)
    net_peak = np.maximum.accumulate(net_equity)
    net_drawdown = (net_peak - net_equity) / net_peak
    max_dd_net = float(np.max(net_drawdown)) if len(net_drawdown) > 0 else 0.0

    gross_equity = np.cumprod(1.0 + gross_returns)
    gross_peak = np.maximum.accumulate(gross_equity)
    gross_drawdown = (gross_peak - gross_equity) / gross_peak
    max_dd_gross = float(np.max(gross_drawdown)) if len(gross_drawdown) > 0 else 0.0

    # Trade statistics
    num_trades = len(trades_df)
    if num_trades > 0:
        wins = int((trades_df["Net_PnL"] > 0).sum())
        win_rate = float(wins / num_trades)
        avg_holding = float(trades_df["Holding_Days"].mean())
        total_costs = float(trades_df["Transaction_Costs"].sum())
        # Turnover: Each trade buys and sells (2 round-trip turns = 2.0 per trade)
        # Annualized turnover = (num_trades * 2.0) / years
        turnover = float((num_trades * 2.0) / years)
    else:
        wins = 0
        win_rate = 0.0
        avg_holding = 0.0
        total_costs = 0.0
        turnover = 0.0

    return {
        "Total_Days": n,
        "Years": round(years, 3),
        "Total_Trades": num_trades,
        "Winning_Trades": wins,
        "Losing_Trades": num_trades - wins,
        "Win_Rate": round(win_rate, 4),
        "Avg_Holding_Days": round(avg_holding, 1),
        "Cumulative_Gross_Return": round(cum_gross, 4),
        "Cumulative_Net_Return": round(cum_net, 4),
        "Annualised_Gross_Return": round(ann_gross_ret, 4),
        "Annualised_Net_Return": round(ann_net_ret, 4),
        "Annualised_Volatility": round(ann_vol, 4),
        "Sharpe_Ratio_rf_zero": round(sharpe_zero, 4),
        "Sharpe_Ratio_rf_6_5pct": round(sharpe_rf, 4),
        "Max_Drawdown_Gross": round(max_dd_gross, 4),
        "Max_Drawdown_Net": round(max_dd_net, 4),
        "Total_Transaction_Costs": round(total_costs, 4),
        "Annualised_Turnover": round(turnover, 2),
    }


def simulate_combined_portfolio(
    pair_daily_dict: Dict[str, pd.DataFrame],
    pair_trades_dict: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    config: Optional[StrategyConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Simulate multi-pair combined portfolio with equal or custom capital allocation.

    Capital Allocation & Overlapping Stock Exposures:
    ------------------------------------------------
    Each pair k is allocated weight w_k (default 1/K).
    Total portfolio capital is normalized to 1.0.
    The portfolio daily return is:
        r_{portfolio, t} = sum_k (w_k * r_{k, t})
    
    Overlapping exposures (e.g. HDFC Bank in Pair 1 and Pair 2):
    Stock exposure is linearly additive:
        Exposure_A,t = sum_k (w_k * Position_{k, t} * w_{A, k})
    This avoids duplicate positions while accurately accounting for net directional risk.

    Args:
        pair_daily_dict: Dictionary mapping pair_name -> daily_df.
        pair_trades_dict: Dictionary mapping pair_name -> trades_df.
        weights: Dictionary mapping pair_name -> float capital weight.
        config: StrategyConfig instance.

    Returns:
        portfolio_daily_df: Combined daily equity and returns.
        combined_trades_df: Consolidated trade log.
        portfolio_metrics: Portfolio performance summary.
    """
    if config is None:
        config = StrategyConfig()

    pairs = list(pair_daily_dict.keys())
    k_pairs = len(pairs)
    if k_pairs == 0:
        raise ValueError("No pair daily data provided for portfolio simulation.")

    if weights is None:
        weights = {p: 1.0 / k_pairs for p in pairs}
    else:
        # Normalize weights
        total_w = sum(weights.values())
        weights = {p: w / total_w for p, w in weights.items()}

    # Merge daily returns
    base_df = list(pair_daily_dict.values())[0][["Date"]].copy()
    
    port_net_ret = np.zeros(len(base_df), dtype=float)
    port_gross_ret = np.zeros(len(base_df), dtype=float)
    port_costs = np.zeros(len(base_df), dtype=float)

    for p in pairs:
        pdf = pair_daily_dict[p]
        w = weights[p]
        port_net_ret += w * pdf["Net_Return"].values
        port_gross_ret += w * pdf["Gross_Return"].values
        port_costs += w * pdf["Transaction_Cost"].values
        base_df[f"Net_Return_{p}"] = pdf["Net_Return"]
        base_df[f"Gross_Return_{p}"] = pdf["Gross_Return"]
        base_df[f"Position_{p}"] = pdf["Position"]
        base_df[f"Net_Equity_{p}"] = pdf["Net_Equity"]

    base_df["Portfolio_Gross_Return"] = port_gross_ret
    base_df["Portfolio_Cost"] = port_costs
    base_df["Portfolio_Net_Return"] = port_net_ret

    base_df["Portfolio_Gross_Equity"] = (1.0 + base_df["Portfolio_Gross_Return"]).cumprod()
    base_df["Portfolio_Net_Equity"] = (1.0 + base_df["Portfolio_Net_Return"]).cumprod()

    base_df["Portfolio_Net_Peak"] = base_df["Portfolio_Net_Equity"].cummax()
    base_df["Portfolio_Net_Drawdown"] = (
        (base_df["Portfolio_Net_Peak"] - base_df["Portfolio_Net_Equity"]) / base_df["Portfolio_Net_Peak"]
    )

    # Consolidated trade log
    trade_frames = []
    for p in pairs:
        tdf = pair_trades_dict[p]
        if not tdf.empty:
            trade_frames.append(tdf)
    
    if trade_frames:
        combined_trades_df = pd.concat(trade_frames, ignore_index=True)
        combined_trades_df.sort_values(by="Entry_Date", inplace=True)
        combined_trades_df.reset_index(drop=True, inplace=True)
        combined_trades_df["Trade_ID"] = np.arange(1, len(combined_trades_df) + 1)
    else:
        combined_trades_df = pd.DataFrame()

    port_metrics = calculate_performance_metrics(
        net_returns=port_net_ret,
        gross_returns=port_gross_ret,
        trades_df=combined_trades_df,
        config=config,
    )
    port_metrics["Pair"] = "COMBINED_PORTFOLIO (Equal Weight)"

    return base_df, combined_trades_df, port_metrics
