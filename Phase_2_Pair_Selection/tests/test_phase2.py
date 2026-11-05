"""
Unit tests for Phase 2: Pair Selection and Cointegration Analysis.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_2_DIR = os.path.dirname(TESTS_DIR)
SCRIPTS_DIR = os.path.join(PHASE_2_DIR, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pair_selection import (
    calculate_correlation_matrix,
    estimate_hedge_ratio,
    test_engle_granger_cointegration,
    estimate_preliminary_ou,
    analyze_all_pairs,
)


class TestPhase2PairSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")

        # Create two cointegrated series: y = 2.0 * x + stationary noise
        x = np.cumsum(np.random.normal(0, 1, n)) + 50.0
        stationary_noise = np.zeros(n)
        for t in range(1, n):
            stationary_noise[t] = 0.85 * stationary_noise[t - 1] + np.random.normal(0, 0.5)
        y = 2.0 * x + stationary_noise

        # Independent random walk z
        z = np.cumsum(np.random.normal(0, 1, n)) + 20.0

        self.df = pd.DataFrame({"STOCK_A": y, "STOCK_B": x, "STOCK_C": z}, index=dates)

    def test_correlation_matrix(self):
        corr = calculate_correlation_matrix(self.df)
        self.assertEqual(corr.shape, (3, 3))
        for col in self.df.columns:
            self.assertAlmostEqual(corr.loc[col, col], 1.0)
        self.assertTrue((corr >= -1.0).all().all())
        self.assertTrue((corr <= 1.0).all().all())

    def test_hedge_ratio_ols(self):
        res = estimate_hedge_ratio(self.df["STOCK_A"], self.df["STOCK_B"])
        # Expected beta is approximately 2.0
        self.assertAlmostEqual(res["beta"], 2.0, delta=0.1)
        self.assertGreater(res["r_squared"], 0.95)
        self.assertEqual(len(res["spread"]), len(self.df))

    def test_coint_on_cointegrated_pair(self):
        coint_res = test_engle_granger_cointegration(
            self.df["STOCK_A"], self.df["STOCK_B"], significance_level=0.05
        )
        # Should detect cointegration
        self.assertTrue(coint_res["is_cointegrated"])
        self.assertLess(coint_res["adf_pvalue"], 0.05)

    def test_coint_on_independent_pair(self):
        coint_res = test_engle_granger_cointegration(
            self.df["STOCK_A"], self.df["STOCK_C"], significance_level=0.05
        )
        # Independent series should not be cointegrated
        self.assertGreater(coint_res["coint_pvalue"], 0.05)

    def test_ou_parameter_estimation(self):
        # Create stationary AR(1) with b = 0.95
        # Expected theta = -252 * ln(0.95) ~= 12.92
        # Expected half-life = ln(2) / -ln(0.95) ~= 13.51 days
        n = 1000
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = 0.95 * spread[t - 1] + np.random.normal(0, 0.1)
        s_series = pd.Series(spread)

        ou_res = estimate_preliminary_ou(s_series)
        self.assertTrue(ou_res["is_valid_ou"])
        self.assertAlmostEqual(ou_res["ar1_b"], 0.95, delta=0.02)
        self.assertAlmostEqual(ou_res["half_life_days"], 13.5, delta=3.0)
        self.assertTrue(ou_res["meets_half_life_criteria"])

    def test_ou_safeguards_unit_root(self):
        # Random walk (b >= 1.0)
        rw = pd.Series(np.cumsum(np.random.normal(0, 1, 500)))
        ou_res = estimate_preliminary_ou(rw)
        # Should flag or produce very high half-life (> 60)
        if not ou_res["is_valid_ou"]:
            self.assertTrue(np.isnan(ou_res["theta_annualized"]))
        else:
            self.assertFalse(ou_res["meets_half_life_criteria"])


if __name__ == "__main__":
    unittest.main()
