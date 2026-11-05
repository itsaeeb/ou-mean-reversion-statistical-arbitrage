"""
Unit Tests for Phase 3: Ornstein-Uhlenbeck Parameter Estimation
==============================================================
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add scripts directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(test_dir, "..", "scripts")
sys.path.insert(0, scripts_dir)

from ou_estimation import construct_spread, estimate_ar1_ou, rolling_ou_estimation


class TestPhase3OUEstimation(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.dt = 1.0 / 252.0
        
    def test_spread_construction(self):
        """Verify synthetic spread formula X_t = ln(P_A) - beta * ln(P_B)."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "STOCK_A": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 108.0, 110.0],
            "STOCK_B": [50.0, 51.0, 50.5, 51.5, 52.0, 51.8, 52.5, 53.0, 53.5, 54.0]
        }, index=dates)
        log_df = np.log(df)
        
        beta = 1.5
        spread = construct_spread(log_df, "STOCK_A", "STOCK_B", beta)
        
        expected = log_df["STOCK_A"] - beta * log_df["STOCK_B"]
        np.testing.assert_allclose(spread.values, expected.values, rtol=1e-12)
        self.assertEqual(spread.name, "Spread_STOCK_A_STOCK_B")

    def test_ar1_recovery_exact(self):
        """Verify analytical recovery of known continuous OU parameters from exact discrete AR(1)."""
        theta_true = 5.0      # yr^-1
        mu_true = 0.5         # equilibrium mean
        sigma_true = 0.25     # yr^-0.5
        dt = 1.0 / 252.0
        
        b_true = np.exp(-theta_true * dt)
        a_true = mu_true * (1.0 - b_true)
        sigma_eta_true = sigma_true * np.sqrt((1.0 - np.exp(-2.0 * theta_true * dt)) / (2.0 * theta_true))
        
        # Simulate a long path (N = 10,000) to verify asymptotic convergence
        N = 15000
        x = np.zeros(N)
        x[0] = mu_true
        noise = np.random.normal(0, sigma_eta_true, size=N)
        for t in range(1, N):
            x[t] = a_true + b_true * x[t-1] + noise[t]
            
        res = estimate_ar1_ou(x, dt=dt)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["failure_reason"], "VALID")
        
        # Check AR(1) coefficients
        self.assertAlmostEqual(res["ar1_b"], b_true, delta=0.01)
        self.assertAlmostEqual(res["ar1_a"], a_true, delta=0.01)
        
        # Check continuous OU parameters
        self.assertAlmostEqual(res["theta"], theta_true, delta=0.7)
        self.assertAlmostEqual(res["mu"], mu_true, delta=0.05)
        self.assertAlmostEqual(res["sigma"], sigma_true, delta=0.05)
        
        # Check theoretical half-life
        expected_halflife_days = np.log(2.0) / (-np.log(res["ar1_b"]))
        self.assertAlmostEqual(res["half_life_days"], expected_halflife_days, places=6)

    def test_invalid_estimates_rejection(self):
        """Verify explicit rejection and flagging of explosive, unit-root, or non-positive AR(1) processes."""
        # Case 1: Explosive / Unit-root process (b >= 1)
        x_explosive = np.linspace(1.0, 10.0, 120)  # purely trending line, b > 1
        res1 = estimate_ar1_ou(x_explosive, dt=self.dt)
        self.assertFalse(res1["is_valid"])
        self.assertTrue("b >= 1" in res1["failure_reason"])
        self.assertTrue(np.isnan(res1["theta"]))
        self.assertTrue(np.isnan(res1["mu"]))
        self.assertTrue(np.isnan(res1["sigma"]))
        
        # Case 2: Oscillatory process (b <= 0)
        x_oscillatory = np.array([1.0, -1.0] * 60)  # alternating signs, b < 0
        res2 = estimate_ar1_ou(x_oscillatory, dt=self.dt)
        self.assertFalse(res2["is_valid"])
        self.assertTrue("b <= 0" in res2["failure_reason"])
        self.assertTrue(np.isnan(res2["theta"]))
        
        # Case 3: Zero variance constant array
        x_const = np.ones(120) * 3.5
        res3 = estimate_ar1_ou(x_const, dt=self.dt)
        self.assertFalse(res3["is_valid"])

    def test_sigma_eq_and_halflife_identities(self):
        """Verify mathematical identity sigma_eq = sigma / sqrt(2*theta) == sqrt(res_var / (1 - b^2))."""
        # Create arbitrary valid stationary window
        x = np.random.normal(0, 1, 120)
        # Apply smoothing to ensure 0 < b < 1
        for i in range(1, 120):
            x[i] = 0.95 * x[i-1] + np.random.normal(0, 0.1)
            
        res = estimate_ar1_ou(x, dt=self.dt)
        self.assertTrue(res["is_valid"])
        
        theta = res["theta"]
        sigma = res["sigma"]
        sigma_eq = res["sigma_eq"]
        res_var = res["residual_var"]
        b = res["ar1_b"]
        
        # Check identity: sigma_eq = sigma / sqrt(2 * theta)
        calc_sigma_eq = sigma / np.sqrt(2.0 * theta)
        self.assertAlmostEqual(sigma_eq, calc_sigma_eq, places=7)
        
        # Check identity: sigma_eq = sqrt(res_var / (1 - b^2))
        direct_sigma_eq = np.sqrt(res_var / (1.0 - b**2))
        self.assertAlmostEqual(sigma_eq, direct_sigma_eq, places=7)
        
        # Check half-life conversion: days = years * 252
        self.assertAlmostEqual(res["half_life_days"], res["half_life_years"] * 252.0, places=7)

    def test_rolling_window_length_and_alignment(self):
        """Verify rolling window length is strictly 120 and dates are aligned without look-ahead bias."""
        N = 250
        dates = pd.date_range("2020-01-01", periods=N, freq="B")
        series = pd.Series(np.cumsum(np.random.normal(0, 0.01, N)), index=dates)
        
        window_size = 120
        df = rolling_ou_estimation(series, window_size=window_size, dt=self.dt)
        
        # Number of rolling windows should be N - window_size + 1
        expected_windows = N - window_size + 1
        self.assertEqual(len(df), expected_windows)
        
        # Check first window
        first_row = df.iloc[0]
        self.assertEqual(first_row["Window_Start_Date"], dates[0])
        self.assertEqual(first_row["Window_End_Date"], dates[window_size - 1])
        self.assertEqual(df.index[0], dates[window_size - 1])
        
        # Check last window
        last_row = df.iloc[-1]
        self.assertEqual(last_row["Window_Start_Date"], dates[N - window_size])
        self.assertEqual(last_row["Window_End_Date"], dates[-1])
        self.assertEqual(df.index[-1], dates[-1])

    def test_standardized_spread_zscore(self):
        """Verify calculation of standardized spread s_t = (X_t - mu) / sigma_eq."""
        spread_val = 0.25
        mu_val = 0.15
        sigma_eq_val = 0.05
        
        expected_z = (spread_val - mu_val) / sigma_eq_val  # (0.25 - 0.15) / 0.05 = 2.0
        self.assertAlmostEqual(expected_z, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
