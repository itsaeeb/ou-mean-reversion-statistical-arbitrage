"""
Unit tests for Phase 1: Data Collection and Preprocessing.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_1_DIR = os.path.dirname(TESTS_DIR)
SCRIPTS_DIR = os.path.join(PHASE_1_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PHASE_1_DIR not in sys.path:
    sys.path.insert(0, PHASE_1_DIR)

from preprocessing import (
    clean_and_align_prices,
    compute_log_prices,
    split_train_test,
    build_data_quality_summary,
)
from data_loader import DEFAULT_TICKERS


class TestPhase1Preprocessing(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset
        dates = pd.date_range("2023-12-28", "2024-01-05", freq="B")
        self.tickers = list(DEFAULT_TICKERS.keys())[:3]
        np.random.seed(42)
        data = np.random.uniform(500, 1500, size=(len(dates), len(self.tickers)))
        self.sample_df = pd.DataFrame(data, index=dates, columns=self.tickers)

    def test_log_price_calculation(self):
        log_df = compute_log_prices(self.sample_df)
        np.testing.assert_allclose(log_df.values, np.log(self.sample_df.values))
        self.assertFalse(np.isneginf(log_df.values).any())
        self.assertFalse(np.isposinf(log_df.values).any())

    def test_log_price_non_positive_error(self):
        bad_df = self.sample_df.copy()
        bad_df.iloc[0, 0] = -5.0
        with self.assertRaises(ValueError):
            compute_log_prices(bad_df)

        bad_df.iloc[0, 0] = 0.0
        with self.assertRaises(ValueError):
            compute_log_prices(bad_df)

    def test_clean_and_align_prices_handles_duplicates(self):
        # Add duplicate row
        dup_df = pd.concat([self.sample_df, self.sample_df.iloc[[0]]]).sort_index()
        self.assertEqual(len(dup_df), len(self.sample_df) + 1)
        cleaned, audit = clean_and_align_prices(dup_df)
        self.assertEqual(len(cleaned), len(self.sample_df))
        self.assertEqual(audit["duplicate_date_count"], 1)

    def test_clean_and_align_prices_drops_unaligned(self):
        # Introduce a NaN
        nan_df = self.sample_df.copy()
        nan_df.iloc[1, 0] = np.nan
        cleaned, audit = clean_and_align_prices(nan_df)
        self.assertEqual(len(cleaned), len(self.sample_df) - 1)
        self.assertEqual(audit["total_missing_values"], 1)
        self.assertEqual(audit["dropped_unaligned_rows"], 1)

    def test_train_test_split(self):
        train, test = split_train_test(
            self.sample_df,
            train_start="2023-12-28",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-01-05",
        )
        self.assertEqual(train.index.max(), pd.to_datetime("2023-12-29"))
        self.assertEqual(test.index.min(), pd.to_datetime("2024-01-01"))
        self.assertEqual(len(train) + len(test), len(self.sample_df))

    def test_summary_metrics(self):
        log_df = compute_log_prices(self.sample_df)
        train, test = split_train_test(
            self.sample_df,
            train_start="2023-12-28",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-01-05",
        )
        summary = build_data_quality_summary(
            self.sample_df, log_df, train, test, DEFAULT_TICKERS
        )
        self.assertEqual(len(summary), len(self.tickers))
        self.assertIn("Annualised Return (%)", summary.columns)
        self.assertIn("Annualised Volatility (%)", summary.columns)


if __name__ == "__main__":
    unittest.main()
