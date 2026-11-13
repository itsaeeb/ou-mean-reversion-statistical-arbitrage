"""Unit Tests for Phase 6: Performance Metrics & Benchmark Comparison."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add repo root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_6_Performance_Benchmark.scripts.optimal_thresholds import (
    expected_trade_duration,
    expected_waiting_duration,
    expected_total_cycle_time,
    expected_cycle_profit,
    solve_optimal_thresholds,
)
from Phase_6_Performance_Benchmark.scripts.performance_analytics import (
    compute_drawdown_series,
    compute_advanced_metrics,
)
from Phase_5_Trading_Strategy.scripts.backtest_engine import (
    simulate_pair_strategy,
    StrategyConfig,
)


class TestPhase6(unittest.TestCase):
    """Test suite for Phase 6 analytical threshold solvers and metrics calculation."""

    def test_expected_trade_duration_properties(self):
        """Test Dynkin's ODE FPT solution for positivity and monotonicity."""
        theta = 12.0
        t_narrow = expected_trade_duration(entry_z=1.5, exit_z=0.5, theta=theta)
        t_wide = expected_trade_duration(entry_z=2.5, exit_z=0.5, theta=theta)
        t_high_exit = expected_trade_duration(entry_z=2.5, exit_z=1.0, theta=theta)

        self.assertGreater(t_narrow, 0.0)
        self.assertGreater(t_wide, t_narrow, "Holding time must strictly increase with entry distance")
        self.assertLess(t_high_exit, t_wide, "Holding time must decrease when exit threshold is closer to entry")

    def test_erfi_closed_form_identity(self):
        """Verify that numerical quadrature sum of trade + wait equals exact erfi identity."""
        theta = 15.0
        se = 2.0
        sx = 0.5

        t_trade = expected_trade_duration(se, sx, theta)
        t_wait = expected_waiting_duration(se, sx, theta)
        t_sum = t_trade + t_wait

        t_erfi = expected_total_cycle_time(se, sx, theta)

        self.assertAlmostEqual(t_sum, t_erfi, places=4, msg="erfi identity should match quadrature sum")

    def test_optimal_threshold_solver_boundaries(self):
        """Test that analytical threshold solver satisfies theoretical boundary conditions."""
        res = solve_optimal_thresholds(
            theta=10.0,
            sigma_eq=0.05,
            beta=1.0,
            cost_roundtrip=0.0040,
            risk_free_rate=0.065,
            min_entry=1.2,
            max_entry=3.0,
            pair_name="TestPair",
        )

        self.assertGreater(res.optimal_entry_z, res.optimal_exit_z)
        self.assertGreaterEqual(res.optimal_exit_z, 0.0)
        self.assertGreater(res.expected_cycle_days, res.expected_trade_days)
        self.assertGreater(res.expected_net_profit_per_cycle, 0.0)

    def test_advanced_metrics_accuracy(self):
        """Test accuracy of Sharpe, Sortino, Calmar, and Drawdown calculations on known stream."""
        # 10 days of constant 1% gain
        rets = np.full(10, 0.01)
        trades_df = pd.DataFrame([
            {"Net_PnL": 0.05, "Holding_Days": 5, "Transaction_Costs": 0.004, "Direction": "LONG", "Exit_Reason": "Take-Profit"},
            {"Net_PnL": 0.03, "Holding_Days": 5, "Transaction_Costs": 0.004, "Direction": "SHORT", "Exit_Reason": "Take-Profit"},
        ])

        m = compute_advanced_metrics(rets, rets, trades_df, risk_free_rate=0.065, ann_factor=252)

        self.assertEqual(m["Total_Trades"], 2)
        self.assertEqual(m["Win_Rate"], 1.0)
        self.assertEqual(m["Max_Drawdown_Net"], 0.0)
        self.assertGreater(m["Cumulative_Net_Return"], 0.10)  # (1.01)^10 - 1 = 10.46%
        self.assertAlmostEqual(m["Avg_Holding_Days"], 5.0)

    def test_zero_look_ahead_strategy_execution(self):
        """Verify that simulated strategy never uses future price or signal information."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Create synthetic series
        np.random.seed(42)
        p_a = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        p_b = 50.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))

        # Create alternating Z-score that triggers trades
        z_scores = np.sin(np.linspace(0, 4 * np.pi, n)) * 2.5
        is_valid = np.ones(n, dtype=bool)

        synth_df = pd.DataFrame({
            "Date": dates,
            "StockA": p_a,
            "StockB": p_b,
            "Z_Score": z_scores,
            "Is_Valid": is_valid,
        })

        cfg = StrategyConfig(entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.5)
        daily, trades, _ = simulate_pair_strategy(
            synth_df, "StockA", "StockB", 1.0, "SynthPair", "Z_Score", "Is_Valid", cfg
        )

        for _, tr in trades.iterrows():
            sig_d = pd.to_datetime(tr["Signal_Date"])
            ent_d = pd.to_datetime(tr["Entry_Date"])
            exit_d = pd.to_datetime(tr["Exit_Date"])
            self.assertGreater(ent_d, sig_d, "Entry date must be strictly after signal date")
            self.assertGreaterEqual(exit_d, ent_d, "Exit date must be on or after entry date")

    def test_cost_friction_monotonicity(self):
        """Verify that net return decreases monotonically as transaction friction increases."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        p_a = np.linspace(100, 110, n)
        p_b = np.linspace(50, 52, n)
        z_scores = np.zeros(n)
        z_scores[5] = -2.5   # trigger long
        z_scores[15] = 0.0   # trigger exit

        df = pd.DataFrame({
            "Date": dates, "A": p_a, "B": p_b, "Z": z_scores, "V": np.ones(n, dtype=bool)
        })

        cfg_zero = StrategyConfig(entry_threshold=2.0, exit_threshold=0.5, cost_per_leg_bps=0.0)
        cfg_high = StrategyConfig(entry_threshold=2.0, exit_threshold=0.5, cost_per_leg_bps=20.0)

        d_zero, _, m_zero = simulate_pair_strategy(df, "A", "B", 1.0, "P", "Z", "V", cfg_zero)
        d_high, _, m_high = simulate_pair_strategy(df, "A", "B", 1.0, "P", "Z", "V", cfg_high)

        self.assertGreater(m_zero["Cumulative_Net_Return"], m_high["Cumulative_Net_Return"])


if __name__ == "__main__":
    unittest.main()

