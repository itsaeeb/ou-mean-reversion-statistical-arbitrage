"""Benchmark Comparison Engine for OU Statistical Arbitrage Strategies.

Executes and contrasts multiple pairs trading configurations strictly out-of-sample:
1. Fixed Heuristic Benchmark:
   - Entry: |s_t| >= 2.0
   - Exit: |s_t| <= 0.5
   - Stop-Loss: |s_t| >= 3.5
2. Analytical Optimal Thresholds:
   - Grounded in the continuous-time OU first-passage-time (FPT) problem (Bertram 2010).
   - Solved analytically from in-sample estimated parameters (theta, sigma_eq, beta).
3. In-Sample Calibrated Strategy:
   - Calibrated via grid sweep strictly on 2019-2023 training data to maximize in-sample Sharpe.
4. Passive Market Benchmarks:
   - Buy-and-Hold Stock A, Buy-and-Hold Stock B, and Equal-Weighted constituent basket.
   - Cash / Risk-Free Rate benchmark (6.5% p.a.).

All backtests enforce strict zero look-ahead bias (signals generated at close of day t,
executed on session t+1) and institutional transaction costs (10 bps/leg = 40 bps round-trip).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from Phase_5_Trading_Strategy.scripts.backtest_engine import (
    simulate_pair_strategy,
    simulate_combined_portfolio,
    StrategyConfig,
)
from Phase_6_Performance_Benchmark.scripts.optimal_thresholds import (
    solve_optimal_thresholds,
    OptimalThresholdResult,
)
from Phase_6_Performance_Benchmark.scripts.performance_analytics import (
    compute_advanced_metrics,
    compute_drawdown_series,
)


@dataclass
class ComparisonRunResult:
    """Container for benchmark comparison outputs."""
    strategy_name: str
    pair_name: str
    config: StrategyConfig
    daily_df: pd.DataFrame
    trades_df: pd.DataFrame
    metrics: Dict[str, Any]


def run_strategy_backtest(
    test_df: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    beta: float,
    pair_name: str,
    z_col: str,
    valid_col: str,
    config: StrategyConfig,
    strategy_name: str,
) -> ComparisonRunResult:
    """Execute strategy backtest and compute advanced performance metrics."""
    daily_df, trades_df, base_metrics = simulate_pair_strategy(
        df=test_df,
        stock_a=stock_a,
        stock_b=stock_b,
        beta=beta,
        pair_name=pair_name,
        z_col=z_col,
        valid_col=valid_col,
        config=config,
    )

    adv_metrics = compute_advanced_metrics(
        net_returns=daily_df["Net_Return"].values,
        gross_returns=daily_df["Gross_Return"].values,
        trades_df=trades_df,
        risk_free_rate=config.risk_free_rate,
        ann_factor=config.annualization_factor,
    )

    adv_metrics["Strategy"] = strategy_name
    adv_metrics["Pair"] = pair_name
    adv_metrics["Entry_Z"] = config.entry_threshold
    adv_metrics["Exit_Z"] = config.exit_threshold
    adv_metrics["Stop_Loss_Z"] = config.stop_loss_threshold
    adv_metrics["Cost_Bps_Per_Leg"] = config.cost_per_leg_bps

    return ComparisonRunResult(
        strategy_name=strategy_name,
        pair_name=pair_name,
        config=config,
        daily_df=daily_df,
        trades_df=trades_df,
        metrics=adv_metrics,
    )


def compute_market_benchmarks(
    test_df: pd.DataFrame,
    risk_free_rate: float = 0.065,
) -> List[Dict[str, Any]]:
    """Compute passive buy-and-hold benchmarks on the out-of-sample period."""
    benchmarks = []
    ann_factor = 252
    n = len(test_df)
    years = n / ann_factor

    banking_tickers = [
        "HDFCBANK.NS", "KOTAKBANK.NS", "SBIN.NS",
        "ICICIBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"
    ]

    # 1. Single Asset Buy & Hold
    for ticker in ["HDFCBANK.NS", "KOTAKBANK.NS", "SBIN.NS"]:
        if ticker in test_df.columns:
            prices = test_df[ticker].values
            rets = np.diff(prices) / prices[:-1]
            rets = np.insert(rets, 0, 0.0)

            empty_trades = pd.DataFrame()
            m = compute_advanced_metrics(rets, rets, empty_trades, risk_free_rate, ann_factor)
            m["Strategy"] = f"Buy & Hold {ticker.replace('.NS', '')}"
            m["Pair"] = "Market Benchmark"
            m["Entry_Z"] = np.nan
            m["Exit_Z"] = np.nan
            m["Stop_Loss_Z"] = np.nan
            m["Cost_Bps_Per_Leg"] = 0.0
            benchmarks.append(m)

    # 2. Equal-Weighted 6-Bank Basket
    available_tickers = [t for t in banking_tickers if t in test_df.columns]
    if available_tickers:
        price_mat = test_df[available_tickers].values
        daily_rets = np.diff(price_mat, axis=0) / price_mat[:-1]
        daily_rets = np.vstack([np.zeros((1, len(available_tickers))), daily_rets])
        basket_rets = np.mean(daily_rets, axis=1)

        empty_trades = pd.DataFrame()
        m = compute_advanced_metrics(basket_rets, basket_rets, empty_trades, risk_free_rate, ann_factor)
        m["Strategy"] = "Equal-Weighted Banking Basket"
        m["Pair"] = "Market Benchmark"
        m["Entry_Z"] = np.nan
        m["Exit_Z"] = np.nan
        m["Stop_Loss_Z"] = np.nan
        m["Cost_Bps_Per_Leg"] = 0.0
        benchmarks.append(m)

    # 3. Cash / Risk-Free Asset (6.5% p.a.)
    cash_rets = np.full(n, (1.0 + risk_free_rate) ** (1.0 / ann_factor) - 1.0)
    cash_rets[0] = 0.0
    empty_trades = pd.DataFrame()
    m = compute_advanced_metrics(cash_rets, cash_rets, empty_trades, risk_free_rate, ann_factor)
    m["Strategy"] = "Cash / Risk-Free Benchmark (6.5%)"
    m["Pair"] = "Cash Benchmark"
    m["Entry_Z"] = np.nan
    m["Exit_Z"] = np.nan
    m["Stop_Loss_Z"] = np.nan
    m["Cost_Bps_Per_Leg"] = 0.0
    benchmarks.append(m)

    return benchmarks


def run_full_benchmark_suite(
    test_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    train_ou_params: Dict[str, Dict[str, float]],
    cost_per_leg_bps: float = 10.0,
    risk_free_rate: float = 0.065,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run full comparative suite across all pairs and strategies.

    Strategies tested:
    1. Fixed Heuristic Benchmark: Entry 2.0, Exit 0.5, Stop 3.5.
    2. Analytical Optimal (OU FPT): Entry s_e*, Exit s_x*, Stop 3.5.
    3. In-Sample Calibrated: Entry 1.75, Exit 0.25, Stop 3.5.

    Returns:
        metrics_df: Table of all performance metrics across strategies.
        all_trades_df: Consolidated trade logs.
        equity_curves_dict: Dictionary of cumulative net equity series.
    """
    all_metrics = []
    all_trades = []
    equity_curves = {"Date": test_df["Date"].values}

    # In-sample calibrated thresholds from train period
    in_sample_thresholds = {
        "HDFCBANK.NS - KOTAKBANK.NS": (1.75, 0.25),
        "HDFCBANK.NS - SBIN.NS": (1.75, 0.25),
        "KOTAKBANK.NS - SBIN.NS": (1.75, 0.25),
    }

    # Store daily returns for multi-pair portfolio aggregation
    strategy_pair_dailies: Dict[str, Dict[str, pd.DataFrame]] = {
        "Fixed Benchmark (2.0 / 0.5)": {},
        "Analytical Optimal (FPT)": {},
        "In-Sample Calibrated (1.75 / 0.25)": {},
    }
    strategy_pair_trades: Dict[str, Dict[str, pd.DataFrame]] = {
        "Fixed Benchmark (2.0 / 0.5)": {},
        "Analytical Optimal (FPT)": {},
        "In-Sample Calibrated (1.75 / 0.25)": {},
    }

    for _, row in pairs_df.iterrows():
        pair = row["Pair"]
        sa = row["Stock A (Y)"]
        sb = row["Stock B (X)"]
        beta = float(row["Hedge Ratio (Beta)"])

        # Recover train OU params
        pair_params = train_ou_params.get(pair, {})
        theta = pair_params.get("Theta", 16.25)
        sigma_eq = pair_params.get("Sigma_eq", 0.0503)

        # 1. Solve Analytical Optimal Thresholds (FPT)
        opt_res = solve_optimal_thresholds(
            theta=theta,
            sigma_eq=sigma_eq,
            beta=beta,
            cost_roundtrip=4.0 * (cost_per_leg_bps / 10000.0),
            risk_free_rate=risk_free_rate,
            min_entry=1.25,
            max_entry=3.0,
            pair_name=pair,
        )

        configs = [
            ("Fixed Benchmark (2.0 / 0.5)", StrategyConfig(
                entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.5,
                cost_per_leg_bps=cost_per_leg_bps, risk_free_rate=risk_free_rate
            )),
            ("Analytical Optimal (FPT)", StrategyConfig(
                entry_threshold=opt_res.optimal_entry_z, exit_threshold=opt_res.optimal_exit_z,
                stop_loss_threshold=3.5, cost_per_leg_bps=cost_per_leg_bps, risk_free_rate=risk_free_rate
            )),
            ("In-Sample Calibrated (1.75 / 0.25)", StrategyConfig(
                entry_threshold=in_sample_thresholds.get(pair, (1.75, 0.25))[0],
                exit_threshold=in_sample_thresholds.get(pair, (1.75, 0.25))[1],
                stop_loss_threshold=3.5, cost_per_leg_bps=cost_per_leg_bps, risk_free_rate=risk_free_rate
            )),
        ]

        for strat_name, cfg in configs:
            run_res = run_strategy_backtest(
                test_df=test_df,
                stock_a=sa,
                stock_b=sb,
                beta=beta,
                pair_name=pair,
                z_col=f"Z_Score_{pair}",
                valid_col=f"Is_Valid_{pair}",
                config=cfg,
                strategy_name=strat_name,
            )

            all_metrics.append(run_res.metrics)
            if not run_res.trades_df.empty:
                tdf = run_res.trades_df.copy()
                tdf["Strategy"] = strat_name
                all_trades.append(tdf)

            col_name = f"{strat_name}__{pair}__Net_Equity"
            equity_curves[col_name] = run_res.daily_df["Net_Equity"].values

            strategy_pair_dailies[strat_name][pair] = run_res.daily_df
            strategy_pair_trades[strat_name][pair] = run_res.trades_df

    # 3. Simulate Combined Multi-Pair Portfolios for each Strategy
    for strat_name in strategy_pair_dailies:
        port_cfg = StrategyConfig(risk_free_rate=risk_free_rate, cost_per_leg_bps=cost_per_leg_bps)
        port_daily, port_trades, port_metrics = simulate_combined_portfolio(
            pair_daily_dict=strategy_pair_dailies[strat_name],
            pair_trades_dict=strategy_pair_trades[strat_name],
            config=port_cfg,
        )

        # Compute full advanced metrics for portfolio
        adv_port_metrics = compute_advanced_metrics(
            net_returns=port_daily["Portfolio_Net_Return"].values,
            gross_returns=port_daily["Portfolio_Gross_Return"].values,
            trades_df=port_trades,
            risk_free_rate=risk_free_rate,
            ann_factor=252,
        )

        adv_port_metrics["Strategy"] = strat_name
        adv_port_metrics["Pair"] = "COMBINED PORTFOLIO"
        adv_port_metrics["Entry_Z"] = np.nan
        adv_port_metrics["Exit_Z"] = np.nan
        adv_port_metrics["Stop_Loss_Z"] = 3.5
        adv_port_metrics["Cost_Bps_Per_Leg"] = cost_per_leg_bps
        all_metrics.append(adv_port_metrics)

        equity_curves[f"{strat_name}__PORTFOLIO__Net_Equity"] = port_daily["Portfolio_Net_Equity"].values

    # 4. Add Market Benchmarks
    market_metrics = compute_market_benchmarks(test_df, risk_free_rate)
    all_metrics.extend(market_metrics)

    # Convert to clean DataFrames
    metrics_df = pd.DataFrame(all_metrics)
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_curves_df = pd.DataFrame(equity_curves)

    return metrics_df, all_trades_df, equity_curves_df
