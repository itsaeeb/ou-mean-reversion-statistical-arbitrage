"""Unit tests for Phase 5 OU Trading Strategy and Backtest.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from Phase_5_Trading_Strategy.scripts.backtest_engine import (
    compute_spread_return,
    simulate_pair_strategy,
    simulate_combined_portfolio,
    calculate_performance_metrics,
    StrategyConfig,
)


class TestPhase5BacktestEngine(unittest.TestCase):
    """Test suite for Phase 5 trading strategy and backtesting modules."""

    def test_compute_spread_return_positive_beta(self):
        """Test gross spread return calculation with positive beta."""
        ret_a = 0.04
        ret_b = 0.02
        beta = 1.0  # Equal weight: (0.04 - 1.0 * 0.02) / (1 + 1) = 0.02 / 2 = 0.01
        res = compute_spread_return(ret_a, ret_b, beta)
        self.assertAlmostEqual(res, 0.01, places=7)

    def test_compute_spread_return_invalid_beta(self):
        """Test error handling for invalid or zero beta."""
        with self.assertRaises(ValueError):
            compute_spread_return(0.01, 0.02, np.nan)
        with self.assertRaises(ValueError):
            compute_spread_return(0.01, 0.02, 0.0)

    def test_strategy_config_costs(self):
        """Test transaction cost calculation properties in StrategyConfig."""
        config = StrategyConfig(cost_per_leg_bps=10.0)
        self.assertAlmostEqual(config.cost_per_leg, 0.0010, places=6)
        self.assertAlmostEqual(config.entry_cost, 0.0020, places=6)
        self.assertAlmostEqual(config.exit_cost, 0.0020, places=6)
        self.assertAlmostEqual(config.round_trip_cost, 0.0040, places=6)

    def test_single_pair_simulation_synthetic(self):
        """Test strategy state machine on controlled synthetic price and spread path."""
        # 5 trading days
        # Day 0: Z = -2.5 (LONG signal generated at close)
        # Day 1: Enters LONG position. Pays 20 bps entry. Z = -1.5 (Hold)
        # Day 2: Z = 0.2 (Crosses exit threshold <= 0.5. Exits at close. Pays 20 bps exit)
        # Day 3: Flat. Z = 0.0
        # Day 4: Flat. Z = 2.2 (SHORT signal generated at close)
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "StockA": [100.0, 102.0, 104.0, 103.0, 103.0],
            "StockB": [100.0, 100.0, 100.0, 100.0, 100.0],
            "Z": [-2.5, -1.5, 0.2, 0.0, 2.2],
            "Valid": [True, True, True, True, True],
        })

        daily_df, trades_df, metrics = simulate_pair_strategy(
            df=df,
            stock_a="StockA",
            stock_b="StockB",
            beta=1.0,
            pair_name="TestPair",
            z_col="Z",
            valid_col="Valid",
        )

        # There should be exactly 1 closed trade (entered day 1, exited day 2)
        self.assertEqual(len(trades_df), 1)
        t = trades_df.iloc[0]
        self.assertEqual(t["Direction"], "LONG")
        self.assertEqual(t["Holding_Days"], 2)
        self.assertEqual(t["Exit_Reason"], "TAKE_PROFIT")
        self.assertAlmostEqual(t["Transaction_Costs"], 0.0040, places=5)

        # Check day 0 signal
        self.assertEqual(daily_df.loc[0, "Signal"], 1)
        # Check day 1 position = 1
        self.assertEqual(daily_df.loc[1, "Position"], 1)
        # Check day 1 cost = 0.0020
        self.assertAlmostEqual(daily_df.loc[1, "Transaction_Cost"], 0.0020, places=5)
        # Check day 2 position = 1 (exit day)
        self.assertEqual(daily_df.loc[2, "Position"], 1)
        # Check day 2 cost = 0.0020 (exit cost)
        self.assertAlmostEqual(daily_df.loc[2, "Transaction_Cost"], 0.0020, places=5)
        # Check day 3 position = 0 (flat)
        self.assertEqual(daily_df.loc[3, "Position"], 0)

    def test_stop_loss_trigger(self):
        """Test emergency stop-loss execution when |s_t| >= 3.5."""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "StockA": [100.0, 98.0, 95.0, 95.0],
            "StockB": [100.0, 100.0, 100.0, 100.0],
            "Z": [-2.2, -2.8, -3.6, -3.0],  # Diverges further to -3.6
            "Valid": [True, True, True, True],
        })

        daily_df, trades_df, metrics = simulate_pair_strategy(
            df=df,
            stock_a="StockA",
            stock_b="StockB",
            beta=1.0,
            pair_name="TestPairSL",
            z_col="Z",
            valid_col="Valid",
        )

        self.assertEqual(len(trades_df), 1)
        t = trades_df.iloc[0]
        self.assertEqual(t["Exit_Reason"], "STOP_LOSS")
        self.assertEqual(t["Holding_Days"], 2)

    def test_combined_portfolio_aggregation(self):
        """Test equal-weighted multi-pair portfolio aggregation."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        p1_daily = pd.DataFrame({
            "Date": dates,
            "Net_Return": [0.01, 0.02, -0.01],
            "Gross_Return": [0.01, 0.02, -0.01],
            "Transaction_Cost": [0.0, 0.0, 0.0],
            "Position": [1, 1, 0],
            "Net_Equity": [1.01, 1.0302, 1.019898],
        })
        p2_daily = pd.DataFrame({
            "Date": dates,
            "Net_Return": [-0.01, 0.00, 0.03],
            "Gross_Return": [-0.01, 0.00, 0.03],
            "Transaction_Cost": [0.0, 0.0, 0.0],
            "Position": [-1, 0, 0],
            "Net_Equity": [0.99, 0.99, 1.0197],
        })

        pair_dict = {"Pair1": p1_daily, "Pair2": p2_daily}
        trades_dict = {"Pair1": pd.DataFrame(), "Pair2": pd.DataFrame()}

        port_df, port_trades, port_metrics = simulate_combined_portfolio(
            pair_daily_dict=pair_dict,
            pair_trades_dict=trades_dict,
        )

        # Equal weight 0.5:
        # Day 0: 0.5*0.01 + 0.5*(-0.01) = 0.00
        # Day 1: 0.5*0.02 + 0.5*(0.00) = 0.01
        # Day 2: 0.5*(-0.01) + 0.5*(0.03) = 0.01
        expected_returns = [0.00, 0.01, 0.01]
        for idx, exp_r in enumerate(expected_returns):
            self.assertAlmostEqual(port_df.loc[idx, "Portfolio_Net_Return"], exp_r, places=6)


if __name__ == "__main__":
    unittest.main()
